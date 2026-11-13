"""Import d'une liste de liens (un par ligne) avec barre de progression.

    python -m webcamdance.import_list liens.txt [--songs-dir songs] [--retry-failed] [--jobs 3]

Pour chaque lien : déjà dans la bibliothèque ? -> sauté. Sinon : téléchargement, coupe de la
partie dansée, extraction et identification des coachs, bandeau, encodage final.
Plusieurs chansons sont traitées en parallèle (--jobs, une par processus) : pendant que l'une
se télécharge, une autre occupe le GPU et une troisième s'encode. Les téléchargements YouTube
restent limités (--max-downloads) pour ne pas déclencher l'anti-robot.
La liste est consommée : doublons, liens déjà importés ou rejetés, et chaque chanson traitée
en sont retirés (sauvegarde .bak au premier lancement ; --keep-list pour ne pas y toucher).
Les échecs y restent pour être retentés. Relançable à volonté (reprend où il en était). Détails dans logs/import_<date>.log, liens en
échec dans logs/import_echecs.txt (réessayables avec --retry-failed).
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace

from . import library
from . import media
from .songinfo import display_name, song_info


def disp_width(s):
    """Largeur affichée dans un terminal (caractères CJK = 2 colonnes)."""
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def fit(s, w):
    """Tronque / complète s à exactement w colonnes affichées."""
    out, n = [], 0
    for c in s:
        cw = disp_width(c)
        if n + cw > w:
            break
        out.append(c)
        n += cw
    return "".join(out) + " " * (w - n)


def fmt_hms(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Progress:
    """Bloc d'état réécrit en place (synthèse + une ligne par import en cours) ; les lignes
    définitives (une par chanson) s'impriment au-dessus. Sortie non-terminal : lignes
    définitives seulement."""

    def __init__(self, total, jobs, out):
        self.total, self.jobs, self.out = total, jobs, out
        self.ok = self.fail = self.rej = self.excl = 0
        self.t0 = time.time()
        self.durations = []
        self.running = {}           # n° -> {"title", "stage", "t0"}
        self.shown = 0              # lignes du bloc actuellement à l'écran
        self.ansi = out.isatty()

    def _clear(self):
        if self.shown:
            self.out.write("\r" + (f"\x1b[{self.shown - 1}A" if self.shown > 1 else "") + "\x1b[J")
            self.shown = 0

    def finish(self):
        """Efface le bloc d'état (fin ou interruption)."""
        self._clear()
        self.out.flush()

    def line(self, msg):
        """Ligne définitive, au-dessus du bloc d'état."""
        w = shutil.get_terminal_size((100, 20)).columns
        self._clear()
        self.out.write(fit(msg, w - 1).rstrip() + "\n")
        self.draw()
        self.out.flush()

    def draw(self):
        if not self.ansi:
            return
        w = shutil.get_terminal_size((100, 20)).columns - 1
        done = self.ok + self.fail + self.rej + self.excl
        frac = done / self.total if self.total else 1.0
        bw = 24
        bar = "█" * int(bw * frac) + "░" * (bw - int(bw * frac))
        eta = ""
        left = self.total - done
        if self.durations and left:
            eta = " · reste ~" + fmt_hms(sum(self.durations) / len(self.durations) * left / min(self.jobs, left))
        lines = [f"[{bar}] {done}/{self.total} · ok {self.ok} · exclues {self.excl} · rejetées {self.rej} · "
                 f"échecs {self.fail}{eta}"]
        now = time.time()
        for k, s in sorted(self.running.items()):
            lines.append(f"  {k}/{self.total} {s['title'][:40]} : {s['stage']} ({fmt_hms(now - s['t0'])})")
        self._clear()
        self.out.write("\n".join(fit(t, w).rstrip() for t in lines))   # jamais plus large que l'écran
        self.shown = len(lines)
        self.out.flush()


# ---------- côté processus d'import (un par import simultané) ----------

class Capture:
    """stdout/stderr du processus d'import : chaque ligne part vers le processus principal
    (journal), et les lignes reconnues mettent à jour l'étape affichée."""

    STAGES = [(r"\[import\] attente d'un créneau", "en attente du téléchargement"),
              (r"\[import\] téléchargement", "téléchargement"),
              (r"\[import\] détection", "recherche de la partie dansée"),
              (r"\[import\] normalisation", "conversion"),
              (r"\[extract\] (\d+) %", "analyse des poses {} %"),
              (r"\[import\] bandeau", "encodage final")]

    def __init__(self, events):
        self.events, self.buf = events, ""
        self.k, self.url, self.root = 0, "", ""

    def write(self, s):
        self.buf += s.replace("\r", "\n")
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if not line.strip():
                continue
            self.events.put(("log", self.k, line))
            if m := re.match(r"\[import\] titre : (.*) ->", line):     # nom propre (corrections, sinon titre)
                name = song_info(m.group(1), library.youtube_id(self.url), self.root, online=False)
                self.events.put(("title", self.k, display_name(name)))
            for pat, label in self.STAGES:
                if m := re.match(pat, line):
                    self.events.put(("stage", self.k, label.format(*m.groups())))
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def _worker_init(events, dl_slots, fs_lock, cookies, native_log, cpu_threads):
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl+C : pris en compte seulement pendant un import
    media.limit_threads(cpu_threads)
    events.cancel_join_thread()                     # sortie du processus jamais bloquée par la file
    fd = os.open(native_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)                                  # sorties natives (CUDA…) : hors de l'écran
    os.dup2(fd, 2)
    sys.stdout = sys.stderr = Capture(events)
    from . import import_song
    import_song.LOCKS.download, import_song.LOCKS.fs = dl_slots, fs_lock
    if cookies:
        media.YT_OPTS["cookiesfrombrowser"] = (cookies,)
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)


def _run(k, url, songs_dir, keep_source):
    """Importe une chanson. -> dict de résultat (jamais d'exception : statut + message)."""
    from .config import Config
    from .import_song import CoachNotDetected, Excluded, import_one
    cap = sys.stdout
    cap.k, cap.url, cap.root = k, url, songs_dir
    t0, last = time.time(), [-1]

    def dl_hook(d):
        tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
        if d.get("status") == "downloading" and tot:
            p = int(100 * d.get("downloaded_bytes", 0) / tot)
            if p != last[0]:
                last[0] = p
                cap.events.put(("stage", k, f"téléchargement {p} %"))

    args = SimpleNamespace(songs_dir=songs_dir, keep_source=keep_source, no_trim=False, start=None,
                           end=None, yes=True, coaches=0, band=None)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        try:
            d = import_one(url, args, Config(), dl_hook=dl_hook)
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            size = sum(f.stat().st_size for f in d.iterdir()) / 1e6
            return {"status": "ok", "meta": m, "size": size, "dt": time.time() - t0}
        except media.YoutubeBlocked as e:
            return {"status": "blocked", "msg": str(e)}
        except Excluded as e:
            return {"status": "excl", "msg": str(e)}
        except CoachNotDetected as e:
            return {"status": "rej", "msg": str(e), "dt": time.time() - t0}
        except Exception as e:                      # vidéo privée / supprimée, réseau, etc.
            msg = str(e).strip().splitlines()[-1][:120] if str(e).strip() else type(e).__name__
            return {"status": "fail", "msg": msg, "repr": repr(e), "dt": time.time() - t0}
    except KeyboardInterrupt:
        return {"status": "stop"}
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cap.write("\n")


# ---------- processus principal ----------

def write_links(path, links):
    """Réécrit le fichier de liens (écriture atomique : pas de fichier tronqué si coupure)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(u + "\n" for u in links), encoding="utf-8")
    tmp.replace(path)


def default_jobs():
    """3 imports simultanés, 1 seul si le GPU a moins de 8 Go (chaque import charge son yolo26x)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return 3 if int(out.split()[0]) >= 8000 else 1
    except Exception:
        return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("links", nargs="?", help="fichier texte de liens (un par ligne)")
    ap.add_argument("--songs-dir", default="songs")
    ap.add_argument("--retry-failed", action="store_true", help="réessayer les liens de logs/import_echecs.txt")
    ap.add_argument("--keep-source", action="store_true")
    ap.add_argument("--keep-list", action="store_true", help="ne pas retirer les liens traités du fichier")
    ap.add_argument("--jobs", type=int, default=0,
                    help="imports simultanés (défaut : 3, ou 1 si le GPU a moins de 8 Go ; 1 = un à la fois)")
    ap.add_argument("--max-downloads", type=int, default=2,
                    help="téléchargements YouTube simultanés au plus (défaut 2 : au-delà, risque d'anti-robot)")
    ap.add_argument("--cookies-from-browser", metavar="NAVIGATEUR",
                    help="utiliser la session YouTube du navigateur (firefox, chrome…) contre le blocage anti-robot")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="threads CPU par import (défaut : moitié des cœurs répartie entre les imports)")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")    # console Windows non UTF-8 : pas de crash sur ✓ ⊘ █
    if sys.platform == "win32":
        os.system("")                           # active les séquences ANSI de la console (bloc d'état)
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    fail_file = logs / "import_echecs.txt"
    src = fail_file if a.retry_failed else Path(a.links or "")
    if not src.is_file():
        sys.exit(f"fichier de liens introuvable : {src}")
    text = src.read_text(encoding="utf-8")
    raw = [t for t in text.split() if t.startswith("http")]
    links = library.parse_links(text)
    done_ids = library.imported_ids(a.songs_dir)
    rejected = set(library.load_rejected(a.songs_dir))
    n_done = sum(library.youtube_id(u) in done_ids for u in links)
    n_rej = sum(library.youtube_id(u) in rejected and library.youtube_id(u) not in done_ids for u in links)
    todo = [u for u in links if library.youtube_id(u) not in done_ids | rejected]
    out = sys.stdout
    consume = not a.keep_list and not a.retry_failed
    if consume:
        bak = src.with_name(src.name + ".bak")
        if not bak.exists():
            shutil.copyfile(src, bak)
        write_links(src, todo)
    print(f"{len(raw)} liens, {len(links)} uniques, {n_done} déjà dans la bibliothèque, {n_rej} rejetée(s) "
          f"précédemment -> {len(todo)} à importer")
    if not todo:
        return
    jobs = max(1, min(a.jobs or default_jobs(), len(todo)))
    log_path = logs / f"import_{time.strftime('%Y%m%d-%H%M%S')}.log"
    native_log = log_path.with_suffix(".natif.log")
    print(f"journal détaillé : {log_path}   ({jobs} import(s) simultané(s) ; Ctrl+C pour arrêter ; "
          f"relancer reprend où ça s'est arrêté)\n")
    prog = Progress(len(todo), jobs, out)
    remaining = list(todo)
    streak = []                                     # derniers messages d'échec consécutifs
    failed = []
    pending = list(enumerate(todo, 1))              # (n°, lien) pas encore lancés
    running = {}                                    # future -> (n°, lien)
    stopping = None                                 # raison d'arrêt : plus rien de nouveau n'est lancé

    cpu_threads = a.cpu_threads or media.default_threads(jobs)
    media.limit_threads(cpu_threads)                # variables d'environnement héritées par les processus
    ctx = mp.get_context("spawn")                   # CUDA : jamais de fork
    events = ctx.Queue()
    pool = ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_worker_init,
                               initargs=(events, ctx.BoundedSemaphore(max(1, a.max_downloads)), ctx.Lock(),
                                         a.cookies_from_browser, str(native_log), cpu_threads))

    def consumed(url):
        if consume:
            remaining.remove(url)
            write_links(src, remaining)

    def drain(log, timeout):
        """Messages des processus d'import : journal, titre et étape affichés."""
        try:
            ev = events.get(timeout=timeout)
            while True:
                kind, k, val = ev
                if kind == "log":
                    log.write(f"[{k}] {val}\n")
                elif k in prog.running:
                    prog.running[k]["title" if kind == "title" else "stage"] = val
                ev = events.get_nowait()
        except queue.Empty:
            pass

    def stop(reason, msg):
        nonlocal stopping
        if stopping:
            return
        stopping = reason
        prog.line(msg)
        if running and reason == "interrupt":
            prog.line(f"   annulation des {len(running)} import(s) en cours (Ctrl+C pour ne pas attendre)…")
        elif running:
            prog.line(f"   fin des {len(running)} import(s) en cours avant l'arrêt (Ctrl+C pour couper)…")

    def handle(r, k, url, log):
        nonlocal streak
        s = prog.running.pop(k, None)
        title = s["title"] if s else url
        st = r["status"]
        if st == "ok":
            m = r["meta"]
            prog.ok += 1
            prog.durations.append(r["dt"])
            consumed(url)
            warn = "  ⚠ coupe à vérifier" if m.get("trim_uncertain") else ""
            prog.line(f"✓ {display_name(m)[:60]} — {fmt_hms(m['duration'])}, "
                      f"{m['n_coaches']} coach(s), {r['size']:.0f} Mo, traité en {fmt_hms(r['dt'])}{warn}")
        elif st == "excl":                          # filtre titre (« China »…) : rapide, pas dans l'ETA
            prog.excl += 1
            consumed(url)
            prog.line(f"– {r['msg']}")
        elif st == "rej":                           # injouable : noté, jamais retenté
            prog.rej += 1
            prog.durations.append(r["dt"])
            consumed(url)
            prog.line(f"⊘ {title[:50]} — rejetée : {r['msg']}")
        elif st == "blocked":                       # tout le reste échouerait : on arrête
            stop("blocked", f"⛔ {r['msg']}")
            return
        elif st == "stop":                          # interrompu (Ctrl+C) : nettoyé, reste dans la liste
            return
        else:
            prog.fail += 1
            prog.durations.append(r.get("dt", 0.0))
            failed.append(url)
            log.write(f"ÉCHEC {url} : {r.get('repr', r['msg'])}\n")
            prog.line(f"✗ {url} — {r['msg']}")
            # même erreur 5 fois de suite = problème général (réseau, YouTube, yt-dlp) : stop
            streak.append(re.sub(r"\[youtube\] [\w-]+:", "", r["msg"]))
            if len(streak) >= 5 and len(set(streak[-5:])) == 1:
                stop("streak", "⛔ 5 échecs identiques de suite : problème général, arrêt. Détails dans le journal.")
            return
        streak = []

    with open(log_path, "w", encoding="utf-8") as log:
        try:
            while True:
                try:
                    while not stopping and pending and len(running) < jobs:
                        k, url = pending.pop(0)
                        known = song_info("", library.youtube_id(url), a.songs_dir, online=False)
                        prog.running[k] = {"title": display_name(known) if known["song"] else url,
                                           "stage": "démarrage", "t0": time.time()}
                        log.write(f"\n===== {k}/{len(todo)} {url}\n")
                        running[pool.submit(_run, k, url, a.songs_dir, a.keep_source)] = (k, url)
                    if not running:
                        break
                    drain(log, 0.25)
                    for f in [f for f in running if f.done()]:
                        k, url = running.pop(f)
                        try:
                            r = f.result()
                        except BrokenProcessPool as e:      # processus d'import tué (mémoire…) : tout s'arrête
                            r = {"status": "fail", "msg": f"processus d'import arrêté brutalement ({e})"}
                            stop("broken", "⛔ un processus d'import s'est arrêté brutalement : arrêt.")
                        except Exception as e:
                            r = {"status": "fail", "msg": f"{type(e).__name__}: {e}"[:120], "repr": repr(e)}
                        handle(r, k, url, log)
                    prog.draw()
                except KeyboardInterrupt:
                    if stopping == "interrupt":     # second Ctrl+C : on n'attend plus
                        raise
                    stopping = None
                    stop("interrupt", "interrompu : relancer la même commande pour reprendre.")
            drain(log, 0)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            prog.finish()
    pool.shutdown(wait=True)
    if native_log.is_file() and native_log.stat().st_size == 0:
        native_log.unlink()
    if failed:
        old = set(library.parse_links(fail_file.read_text(encoding="utf-8"))) if fail_file.is_file() else set()
        fail_file.write_text("\n".join(sorted(old | set(failed))) + "\n", encoding="utf-8")
    elif a.retry_failed and not stopping:
        fail_file.unlink(missing_ok=True)
    print(f"\nterminé en {fmt_hms(time.time() - prog.t0)} : {prog.ok} importée(s), {prog.excl} exclue(s), "
          f"{prog.rej} rejetée(s), "
          f"{prog.fail} échec(s)"
          + (f" (liste : {fail_file}, réessayer avec --retry-failed)" if failed else ""))


if __name__ == "__main__":
    main()
