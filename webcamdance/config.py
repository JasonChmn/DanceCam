from dataclasses import dataclass


@dataclass
class Config:
    # --- modèles (repli auto yolo26 -> yolo11 si poids introuvables) ---
    rt_model: str = "yolo26s-pose.pt"       # temps réel ; passer un .engine TensorRT si exporté
    offline_model: str = "yolo26x-pose.pt"  # extraction coach, pas de contrainte de latence
    imgsz: int = 640
    det_conf: float = 0.35
    kpt_conf: float = 0.35                  # seuil de visibilité d'un keypoint
    # --- média ---
    video_height: int = 720                 # net en plein écran 1080p (bandeau ~700 px de haut)
    video_crf: int = 24                     # H.264 ; compromis netteté / disque (~30-35 Mo/chanson recadrée)
    exclude_title: str = r"(?i)\bchina\b"   # titres exclus de l'import (regex, "" = aucun)
    min_coach_presence: float = 0.70        # présence moyenne des coachs sous laquelle l'import est rejeté
    band_margin: float = 0.05               # marge gardée autour du bandeau coach à l'encodage final
    video_fps: int = 30
    render_fps: int = 30                    # vidéo et webcam sont à 30 : plus haut = HUD plus fluide
    game_threads: int = 4                   # threads CPU en jeu (OpenCV, PyTorch) : 4 suffisent pour 30 i/s
    # --- caméra ---
    cam_index: int = 0
    cam_w: int = 1280
    cam_h: int = 720
    cam_fps: int = 30
    cam_latency: float = 0.08               # s, capture->horodatage (à calibrer)
    # --- tracking joueurs ---
    lane_weight: float = 1.5                # poids du prior couloir (joueurs)
    # --- scoring ---
    theta_max_deg: float = 60.0             # erreur angulaire donnant 0
    win_back: float = 0.30                  # tolérance joueur en retard (s)
    win_fwd: float = 0.15                   # tolérance joueur en avance (s)
    static_weight: float = 0.3              # poids relatif d'un os immobile chez le coach
    omega0: float = 2.0                     # rad/s, normalisation du poids de mouvement
    player_min_cover: float = 0.5           # part (pondérée) des os visibles du coach que le joueur doit montrer
    judge_min_frames: int = 5               # frames jugeables mini par move, sinon move « non jugé »
    # (os visible chez le coach mais pas chez le joueur : ignoré, pas pénalisé — occlusion, détection ratée)
    min_cover: float = 0.4                  # fraction d'os coach visibles pour juger
    move_period: float = 0.75               # durée d'un "move" jugé (s)
    grace: float = 0.25                     # attente des poses en retard avant jugement (s)
    # (seuil, label, points). Calé sur le bot (2026-11) : imitation parfaite p10 0.80 / médiane 0.88,
    # danse fausse ou immobile médiane ~0.43 / p90 ~0.55. -> parfait ~94 % PERFECT, faux ~74 % MISS.
    # À affiner avec de vraies parties (un humain imite moins fidèlement que le bot).
    judgements: tuple = ((0.78, "PERFECT", 100), (0.68, "SUPER", 75),
                         (0.58, "GOOD", 50), (0.48, "OK", 25), (0.0, "MISS", 0))
