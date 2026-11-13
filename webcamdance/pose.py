"""Wrapper Ultralytics YOLO-pose."""
import numpy as np

_EMPTY = (np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


class PoseEstimator:
    def __init__(self, weights, cfg):
        import torch
        from ultralytics import YOLO
        self.cfg = cfg
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"
        if self.device == "cpu":
            print("[pose] ATTENTION : CUDA indisponible, inférence CPU.")
        try:
            self.model = YOLO(weights)
        except Exception as e:  # poids yolo26 absents de la version installée
            alt = weights.replace("yolo26", "yolo11")
            if alt == weights:
                raise
            print(f"[pose] {weights} indisponible ({e}); repli sur {alt}")
            self.model = YOLO(alt)
        # Ultralytics récents : 'half' déprécié (warning à chaque appel) au profit de 'quantize'
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            self.prec = {"quantize": 16} if self.half else {}
        else:
            self.prec = {"half": self.half}
        self(np.zeros((480, 640, 3), np.uint8))  # warmup

    def __call__(self, frames):
        """frame BGR ou liste -> liste de (kpts (M,17,3), boxes xyxy (M,4), conf (M,))."""
        if isinstance(frames, np.ndarray):
            frames = [frames]
        res = self.model.predict(frames, imgsz=self.cfg.imgsz, conf=self.cfg.det_conf,
                                 device=self.device, verbose=False, **self.prec)
        out = []
        for r in res:
            if r.keypoints is None or len(r.boxes) == 0:
                out.append(_EMPTY)
                continue
            out.append((r.keypoints.data.float().cpu().numpy(),
                        r.boxes.xyxy.float().cpu().numpy(),
                        r.boxes.conf.float().cpu().numpy()))
        return out
