from __future__ import annotations
import json

def load_coco(path:str):
    with open(path,'r',encoding='utf-8') as f: return json.load(f)
