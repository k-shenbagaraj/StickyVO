from omegaconf import OmegaConf
from .utils import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container

class TwoViewPipeline(BaseModel):
    """
    TwoViewPipeline coordinates the feature extraction, matching, and optional 
    tracking propagation between a pair of views.
    """
    default_conf = {
        "extractor": {"name": None, "trainable": False},
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
        "track_propagation": {
            "enabled": True,
            "matches_key": "matches0",
            "unmatched_value": -1,
            "prev_ids_cache_key": "track_ids",
            "kpts_key_base": "keypoints",
            "out_ids0": "track_ids0",
            "out_ids1": "track_ids1",
        },
    }

    required_data_keys = ["view0", "view1"]
    strict_conf = False 
    components = ["extractor", "matcher", "filter", "solver", "ground_truth"]

    # --- Lifecycle ---

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))
        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))
        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))
        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))
        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(to_ctr(conf.ground_truth))

        self._next_track_id = 0

    # --- Track ID Utilities ---

    def _to_int(self, x):
        import numpy as np
        try:
            import torch
            if torch.is_tensor(x):
                return int(x.item()) if x.numel() == 1 else int(x.reshape(-1)[0].item())
        except Exception: pass

        if isinstance(x, (int, np.integer)): return int(x)
        try:
            arr = np.asarray(x)
            if arr.size >= 1: return int(arr.reshape(-1)[0])
        except Exception: pass
        if hasattr(x, "item"):
            try: return int(x.item())
            except Exception: pass
        return int(x)

    def _normalize_ids(self, ids):
        import numpy as np
        try:
            import torch
            if torch.is_tensor(ids): ids = ids.reshape(-1).detach().cpu().tolist()
        except Exception: pass
        if isinstance(ids, np.ndarray): ids = ids.reshape(-1).tolist()
        if isinstance(ids, (int, np.integer)): return [int(ids)]
        return [self._to_int(v) for v in ids]

    def _bump_next_id_above(self, used_ids):
        if used_ids is None: return
        m = -1
        for tid in used_ids:
            if tid is None: continue
            t = self._to_int(tid)
            if t >= 0: m = max(m, t)
        if m >= 0: self._next_track_id = max(self._next_track_id, m + 1)

    def _alloc_new_ids(self, n):
        ids = list(range(self._next_track_id, self._next_track_id + n))
        self._next_track_id += n
        return ids

    def _infer_len(self, obj):
        if obj is None: return 0
        try: return len(obj)
        except Exception:
            try: return int(obj.shape[0])
            except Exception: return 0

    # --- ID Propagation ---

    def _get_prev_track_ids(self, pred, data):
        tp = self.conf.track_propagation
        cache_key, out_ids0 = tp.prev_ids_cache_key, tp.out_ids0
        view0_cache = (data.get("view0", {}) or {}).get("cache", {}) or {}
        
        if cache_key in view0_cache: return self._normalize_ids(view0_cache[cache_key])
        if out_ids0 in pred: return self._normalize_ids(pred[out_ids0])

        n0 = self._infer_len(pred.get(tp.kpts_key_base + "0"))
        return self._alloc_new_ids(n0) if n0 > 0 else None

    def _propagate_track_ids(self, pred, data):
        tp = self.conf.track_propagation
        if not tp.enabled: return pred

        kb = tp.kpts_key_base
        n0, n1 = self._infer_len(pred.get(kb+"0")), self._infer_len(pred.get(kb+"1"))
        if n0 <= 0 or n1 <= 0: return pred

        prev_ids0 = self._get_prev_track_ids(pred, data)
        if prev_ids0 is None: return pred
        self._bump_next_id_above(prev_ids0)

        unmatched = self._to_int(tp.unmatched_value)
        ids1 = [unmatched] * n1
        matches_key = tp.matches_key

        if matches_key == "matches0":
            m0 = pred.get("matches0", pred.get(matches_key))
            if m0 is not None:
                for i in range(min(n0, self._infer_len(m0))):
                    j = self._to_int(m0[i])
                    if 0 <= j < n1: ids1[j] = self._to_int(prev_ids0[i])
        elif matches_key == "matches01":
            pairs = pred.get("matches01", pred.get(matches_key))
            if pairs is not None:
                for i, j in pairs:
                    ii, jj = self._to_int(i), self._to_int(j)
                    if 0 <= ii < n0 and 0 <= jj < n1: ids1[jj] = self._to_int(prev_ids0[ii])
        else:
            gen = pred.get(matches_key)
            if gen is not None and self._infer_len(gen) == n0:
                for i in range(min(n0, self._infer_len(gen))):
                    j = self._to_int(gen[i])
                    if 0 <= j < n1: ids1[j] = self._to_int(prev_ids0[i])

        new_ids = self._alloc_new_ids(sum(1 for i in ids1 if self._to_int(i) == unmatched))
        ni_it = iter(new_ids)
        for i, tid in enumerate(ids1):
            if self._to_int(tid) == unmatched: ids1[i] = next(ni_it)

        pred[tp.out_ids0] = self._normalize_ids(prev_ids0)
        pred[tp.out_ids1] = self._normalize_ids(ids1)
        return pred

    # --- Forward Pipeline ---

    def extract_view(self, data, i):
        d_i = data[f"view{i}"]
        p_i = d_i.get("cache", {})
        if not (len(p_i) > 0 and self.conf.allow_no_extract) and self.conf.extractor.name:
            p_i = {**p_i, **self.extractor({**d_i, **p_i} if not self.conf.allow_no_extract else d_i)}
        return p_i

    def _forward(self, data):
        pred0, pred1 = self.extract_view(data, "0"), self.extract_view(data, "1")
        pred = {**{k+"0": v for k,v in pred0.items()}, **{k+"1": v for k,v in pred1.items()}}

        if self.conf.matcher.name: pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name: pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name: pred = {**pred, **self.solver({**data, **pred})}

        pred = self._propagate_track_ids(pred, data)

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self, pred, data):
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        losses, metrics, total = {}, {}, 0
        for k in self.components:
            apply = self.conf[k].get("apply_loss", True)
            if self.conf[k].name and apply:
                try: l_, m_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError: continue
                losses, metrics, total = {**losses, **l_}, {**metrics, **m_}, l_["total"] + total
        return {**losses, "total": total}, metrics
