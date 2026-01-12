from typing import Dict
import numpy as np
import torch
from lightgluestick.utils import batch_to_np, numpy_image_to_torch
from lightgluestick.two_view_pipeline import TwoViewPipeline

_LGS_PIPELINE = None
_LGS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LGS_CONF_DEPTH = None
_LGS_LAST_VIEW_CACHE = None

# Tracking state 
_LGS_LAST_TRACK_IDS_PTS = None     
_LGS_NEXT_TRACK_ID_PTS = 0
_LGS_LAST_TRACK_IDS_LINES = None  
_LGS_NEXT_TRACK_ID_LINES = 0

# --- Pipeline Management ---

def _get_lgs_pipeline(depth_confidence: float) -> TwoViewPipeline:
    global _LGS_PIPELINE, _LGS_CONF_DEPTH
    if _LGS_PIPELINE is not None and _LGS_CONF_DEPTH == depth_confidence:
        return _LGS_PIPELINE

    conf = {
        "name": "two_view_pipeline",
        "use_lines": True,
        "allow_no_extract": True,
        "extractor": {
            "name": "wireframe",
            "point_extractor": {
                "name": "superpoint",
                "trainable": False,
                "dense_outputs": True,
                "max_num_keypoints": 2048,
            },
            "line_extractor": {
                "name": "lsd",
                "trainable": False,
                "max_num_lines": 250,
                "min_length": 15,
            },
            "wireframe_params": {"merge_points": True, "merge_line_endpoints": True, "nms_radius": 3},
        },
        "matcher": {"name": "lightgluestick", "depth_confidence": depth_confidence, "trainable": False},
        "ground_truth": {"from_pose_depth": False},
    }

    _LGS_PIPELINE = TwoViewPipeline(conf).to(_LGS_DEVICE).eval()
    _LGS_CONF_DEPTH = depth_confidence
    return _LGS_PIPELINE

# --- Geometric & Matching Helpers ---

def _normalize_lines(arr: np.ndarray) -> np.ndarray:
    if arr is None: return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1:] == (2, 2): arr = arr.reshape(arr.shape[0], 4)
    elif not (arr.ndim == 2 and arr.shape[1] == 4):
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] < 4: return np.zeros((0, 4), dtype=np.float32)
        arr = arr[:, :4]
    return arr.astype(np.float32)

def _propagate_ids_from_matches0(prev_ids0: list, matches0: np.ndarray, n1: int, next_id_start: int) -> (list, int):
    ids1 = [-1] * n1
    for i0 in range(min(len(prev_ids0), int(matches0.shape[0]))):
        j1 = int(matches0[i0])
        if 0 <= j1 < n1: ids1[j1] = int(prev_ids0[i0])
    
    next_id = int(next_id_start)
    for j1 in range(n1):
        if ids1[j1] == -1:
            ids1[j1] = next_id
            next_id += 1
    return ids1, next_id

def _matches_as_index_pairs(matches0: np.ndarray) -> np.ndarray:
    valid = matches0 != -1
    idx0 = np.where(valid)[0].astype(np.int32)
    idx1 = matches0[valid].astype(np.int32)
    if idx0.size == 0: return np.empty((0, 2), dtype=np.int32)
    return np.stack([idx0, idx1], axis=1)

def reset_lgs_sequence():
    """Reset persistent track IDs for a new sequence."""
    global _LGS_LAST_VIEW_CACHE, _LGS_LAST_TRACK_IDS_PTS, _LGS_NEXT_TRACK_ID_PTS, _LGS_LAST_TRACK_IDS_LINES, _LGS_NEXT_TRACK_ID_LINES
    _LGS_LAST_VIEW_CACHE = None
    _LGS_LAST_TRACK_IDS_PTS = None
    _LGS_NEXT_TRACK_ID_PTS = 0
    _LGS_LAST_TRACK_IDS_LINES = None
    _LGS_NEXT_TRACK_ID_LINES = 0

def shutdown_lgs():
    """Explicitly release GPU resources and clear cache."""
    global _LGS_PIPELINE, _LGS_LAST_VIEW_CACHE
    _LGS_PIPELINE = None
    _LGS_LAST_VIEW_CACHE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Core Inference Logic ---

def _run_lightgluestick_internal(img0_gray: np.ndarray, img1_gray: np.ndarray, depth_confidence: float = -1.0, sequential: bool = True) -> Dict[str, np.ndarray]:
    global _LGS_LAST_VIEW_CACHE, _LGS_LAST_TRACK_IDS_PTS, _LGS_NEXT_TRACK_ID_PTS, _LGS_LAST_TRACK_IDS_LINES, _LGS_NEXT_TRACK_ID_LINES
    
    pipeline_model = _get_lgs_pipeline(depth_confidence)
    tg0 = numpy_image_to_torch(img0_gray).to(_LGS_DEVICE)[None]
    tg1 = numpy_image_to_torch(img1_gray).to(_LGS_DEVICE)[None]

    x = {"view0": {"image": tg0}, "view1": {"image": tg1}}
    if sequential and _LGS_LAST_VIEW_CACHE is not None:
        x["view0"]["cache"] = dict(_LGS_LAST_VIEW_CACHE)

    with torch.no_grad(): pred = pipeline_model(x)

    # Cache view1 for subsequent calls
    _LGS_LAST_VIEW_CACHE = {k[:-1]: v for k, v in pred.items() if k.endswith("1")}
    pred_np = batch_to_np(pred)

    kp0, kp1 = pred_np["keypoints0"], pred_np["keypoints1"]
    m0_pts = pred_np["matches0"]
    ls0 = _normalize_lines(pred_np.get("lines0"))
    ls1 = _normalize_lines(pred_np.get("lines1"))
    m0_lines = pred_np.get("line_matches0", np.full((ls0.shape[0],), -1, dtype=np.int32))

    # Propagate Track IDs for points
    n0p, n1p = int(kp0.shape[0]), int(kp1.shape[0])
    if _LGS_LAST_TRACK_IDS_PTS is None or len(_LGS_LAST_TRACK_IDS_PTS) != n0p:
        _LGS_LAST_TRACK_IDS_PTS = list(range(_LGS_NEXT_TRACK_ID_PTS, _LGS_NEXT_TRACK_ID_PTS + n0p))
        _LGS_NEXT_TRACK_ID_PTS += n0p
    _LGS_LAST_TRACK_IDS_PTS, _LGS_NEXT_TRACK_ID_PTS = _propagate_ids_from_matches0(_LGS_LAST_TRACK_IDS_PTS, m0_pts, n1p, _LGS_NEXT_TRACK_ID_PTS)

    # Propagate Track IDs for lines
    n0l, n1l = int(ls0.shape[0]), int(ls1.shape[0])
    if _LGS_LAST_TRACK_IDS_LINES is None or len(_LGS_LAST_TRACK_IDS_LINES) != n0l:
        _LGS_LAST_TRACK_IDS_LINES = list(range(_LGS_NEXT_TRACK_ID_LINES, _LGS_NEXT_TRACK_ID_LINES + n0l))
        _LGS_NEXT_TRACK_ID_LINES += n0l
    _LGS_LAST_TRACK_IDS_LINES, _LGS_NEXT_TRACK_ID_LINES = _propagate_ids_from_matches0(_LGS_LAST_TRACK_IDS_LINES, m0_lines, n1l, _LGS_NEXT_TRACK_ID_LINES)

    return {
        "kpts0": kp0.astype(np.float32, copy=False), "kpts1": kp1.astype(np.float32, copy=False),
        "lines0": ls0.astype(np.float32, copy=False), "lines1": ls1.astype(np.float32, copy=False),
        "matches_points": _matches_as_index_pairs(m0_pts),
        "matches_lines": _matches_as_index_pairs(m0_lines),
    }

# --- Entry Point (C++ Interface) ---

def infer_pair(img0, img1, K: dict = None, cfg: dict = None) -> Dict[str, np.ndarray]:
    """Primary entry point for C++ lgs_frontend."""
    depth_conf = float((cfg or {}).get("depth_confidence", -1.0))

    def to_gray(img):
        if img.ndim == 3: return (0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]).astype(np.uint8)
        return img.astype(np.uint8, copy=False)

    res = _run_lightgluestick_internal(to_gray(img0), to_gray(img1), depth_conf, sequential=True)
    return {
        "kpts0": res["kpts0"], "kpts1": res["kpts1"],
        "lines0": res["lines0"], "lines1": res["lines1"],
        "pt_matches": res["matches_points"].astype(np.int32, copy=False),
        "line_matches": res["matches_lines"].astype(np.int32, copy=False),
    }
