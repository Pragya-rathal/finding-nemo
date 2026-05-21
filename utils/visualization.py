from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

def draw_detections(image: np.ndarray, dets: List[Dict], out_path: str) -> None:
    canvas=image.copy()
    for d in dets:
        x1,y1,x2,y2=map(int,d['bbox'])
        txt=f"{d['label']}:{d['score']:.2f}"
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(canvas,txt,(x1,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, canvas)
