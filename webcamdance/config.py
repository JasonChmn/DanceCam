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
    render_fps: int = 60                    # images/s max de l'affichage (vidéo à 30 : chaque image montrée 2 fois)
    vsync: bool = True                      # synchro écran : images régulières, sans saccade ni déchirure
    show_pictos: bool = True                # bandeau des mouvements à venir (touche P en jeu)
    pause_on_focus_loss: bool = True        # pause auto si la fenêtre perd le focus (Alt+Tab, notification)
    game_threads: int = 4                   # threads CPU en jeu (OpenCV, PyTorch) : 4 suffisent pour 30 i/s
    # --- caméra ---
    cam_index: int = 0
    cam_w: int = 1280
    cam_h: int = 720
    cam_fps: int = 30
    cam_latency: float = 0.08               # s, capture->horodatage (à calibrer)
    # --- tracking joueurs ---
    lane_weight: float = 1.5                # poids du prior couloir (joueurs)
    # --- mode groupe (5+ joueurs, plusieurs rangées) ---
    group_max: int = 8                      # joueurs max (une couleur chacun)
    group_imgsz: int = 960                  # entrée YOLO plus grande : joueurs du fond petits / à moitié cachés
    group_min_cover: float = 0.25           # on note ce qui est visible, même si seuls les bras dépassent
    group_combo_frac: float = 0.7           # part des joueurs jugés en PERFECT/SUPER pour un combo d'équipe
    # --- scoring ---
    theta_max_deg: float = 60.0             # erreur angulaire donnant 0
    win_back: float = 0.60                  # tolérance joueur en retard (s) ; mesuré : ~0,45 s réel (latence + réaction)
    win_fwd: float = 0.15                   # tolérance joueur en avance (s)
    static_weight: float = 0.3              # poids relatif d'un os immobile chez le coach
    omega0: float = 2.0                     # rad/s, normalisation du poids de mouvement
    player_min_cover: float = 0.5           # part (pondérée) des os visibles du coach que le joueur doit montrer
    judge_min_frames: int = 5               # frames jugeables mini par move, sinon move « non jugé »
    # (os visible chez le coach mais pas chez le joueur : ignoré, pas pénalisé — occlusion, détection ratée)
    side_fix: bool = True                   # joueur vu de biais/profil : perspective, bras caché, gauche/droite
    side_gate: float = 0.6                  # épaules joueur / épaules coach (rapportées au tronc) sous lequel on corrige
    side_hide: float = 0.5                  # sous ce rapport (~60°), le bras le moins sûr est ignoré (deviné)
    side_stretch: float = 2.0               # élargissement horizontal maximal
    min_cover: float = 0.4                  # fraction d'os coach visibles pour juger
    move_period: float = 0.75               # durée d'un "move" jugé (s)
    grace: float = 0.25                     # attente des poses en retard avant jugement (s)
    # (seuil, label, points). Calé sur le bot (2026-11) : imitation parfaite p10 0.80 / médiane 0.88,
    # danse fausse ou immobile médiane ~0.43 / p90 ~0.55. -> parfait ~94 % PERFECT, faux ~74 % MISS.
    # À affiner avec de vraies parties (un humain imite moins fidèlement que le bot).
    # 2026-11-13 : seuils baissés de 0,03 après une vraie partie (humain : PERFECT 19 % -> 40 %,
    # danse fausse toujours ~35 % MISS / 0 % PERFECT).
    judgements: tuple = ((0.75, "PERFECT", 100), (0.65, "SUPER", 75),
                         (0.55, "GOOD", 50), (0.45, "OK", 25), (0.0, "MISS", 0))
