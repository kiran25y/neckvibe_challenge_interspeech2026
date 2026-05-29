import os
import glob
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.io import loadmat
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import importlib.util


# =======================
# CONFIG: Change according to your path
# =======================
TEST_FEATURE_DIR   = "/workspace/NeckVibeChallenge/Test_set/Test_features"
TEST_INDEX_CSV     = "/workspace/NeckVibeChallenge/Test_set/Test_features_index.csv"

TEMPLATE_PVH  = "/workspace/NeckVibeChallenge/Test_set/Test_results_template/Task1_PVH_detection.csv"
TEMPLATE_NPVH = "/workspace/NeckVibeChallenge/Test_set/Test_results_template/Task2_NPVH_detection.csv"

OUT_DIR = "/workspace/final_neckvibe/submission_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# PVH tabular
PVH_TAB_MODEL = "/workspace/final_neckvibe/pvh_out/pvh_catboost.cbm"
PVH_TAB_FEATS = "/workspace/final_neckvibe/pvh_out/pvh_features.json"

# NPVH tabular
NPVH_TAB_MODEL = "/workspace/final_neckvibe/npvh_out_final/npvh_catboost_final.cbm"
NPVH_TAB_FEATS = "/workspace/final_neckvibe/npvh_out_final/npvh_features_final.json"

# MIL checkpoints 
PVH_MIL_DIR  = "/workspace/final_neckvibe/mil_pvh_enhanced_out"
NPVH_MIL_DIR = "/workspace/final_neckvibe/mil_npvh_out_improved"

# IMPORTANT: point to your actual MIL training scripts 
PVH_MIL_PY  = "/workspace/final_neckvibe/mil_pvh.py"
NPVH_MIL_PY = "/workspace/final_neckvibe/mil_npvh.py"

# stacking ensemble artifacts
PVH_ENS_DIR  = "/workspace/final_neckvibe/pvh_ensemble_final"
NPVH_ENS_DIR = "/workspace/final_neckvibe/npvh_ensemble_final"

PVH_META_PKL   = os.path.join(PVH_ENS_DIR,  "meta_model.pkl")
PVH_QT_TAB_PKL  = os.path.join(PVH_ENS_DIR,  "qt_tab.pkl")
PVH_QT_MIL_PKL  = os.path.join(PVH_ENS_DIR,  "qt_mil.pkl")

NPVH_META_PKL  = os.path.join(NPVH_ENS_DIR, "meta_model.pkl")
NPVH_QT_TAB_PKL = os.path.join(NPVH_ENS_DIR, "qt_tab.pkl")
NPVH_QT_MIL_PKL = os.path.join(NPVH_ENS_DIR, "qt_mil.pkl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
USE_AMP = (DEVICE == "cuda")

THRESH = 0.5


# =======================
# Helpers
# =======================
def safe_read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def clip01(p):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, 0.0, 1.0)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_subject_id_from_mat(filename: str) -> str:
    base = os.path.basename(filename)
    if "_" in base:
        return base.split("_")[0]
    return base.split(".")[0]

def build_subject_to_paths(test_feature_dir: str, test_index_csv: str = None):
    subject_to_paths = {}

    if test_index_csv and os.path.exists(test_index_csv):
        idx = pd.read_csv(test_index_csv)

        # pick filename column
        fcol = None
        for c in idx.columns:
            cl = c.lower()
            if ("file" in cl or "feature" in cl) and ("mat" in cl or "filename" in cl):
                fcol = c
                break
        if fcol is None:
            fcol = idx.columns[0]

        for fn in idx[fcol].astype(str).tolist():
            fn = fn.strip()
            if not fn or fn.lower() in ["nan", "none"]:
                continue
            mat_path = fn if os.path.isabs(fn) else os.path.join(test_feature_dir, fn)
            if os.path.exists(mat_path):
                sid = parse_subject_id_from_mat(mat_path)
                subject_to_paths.setdefault(sid, []).append(mat_path)
    else:
        mats = sorted(glob.glob(os.path.join(test_feature_dir, "*.mat")))
        for mp in mats:
            sid = parse_subject_id_from_mat(mp)
            subject_to_paths.setdefault(sid, []).append(mp)

    for sid in list(subject_to_paths.keys()):
        subject_to_paths[sid] = sorted(subject_to_paths[sid])

    return subject_to_paths


# =======================
# TABULAR FEATURE EXTRACTION
# =======================
PRIMARY_KEYS = [
    "H1H2all", "LHratioall", "cppall", "dBcms2", "spectralTiltall", "level",
    "IBIF_h1h2", "IBIF_hrf", "IBIF_mfdr", "IBIF_acflow", "IBIF_sq",
    "IBIF_oq", "IBIF_naq", "IBIF_cq",
    "fo",
]

def _safe_float(x):
    x = np.asarray(x).astype(np.float32).squeeze()
    x = x[np.isfinite(x)]
    return x

def mad(x):
    x = _safe_float(x)
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))

def iqr(x):
    x = _safe_float(x)
    if x.size == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def entropy_hist(x, bins=32):
    x = _safe_float(x)
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    p = hist / np.sum(hist)
    return float(-np.sum(p * np.log(p + 1e-12)))

def gini(x):
    x = _safe_float(x)
    if x.size == 0:
        return np.nan
    x = np.abs(x)
    s = np.sum(x)
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * s))

def robust_burst_rate(x):
    x = _safe_float(x)
    if x.size == 0:
        return np.nan
    thr = np.median(x) + 2.0 * mad(x)
    return float(np.mean(x > thr))

def stat_pack(x):
    x = _safe_float(x)
    if x.size < 5:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "iqr": np.nan, "mad": np.nan,
            "skew": np.nan, "kurt": np.nan,
            "p01": np.nan, "p05": np.nan, "p95": np.nan, "p99": np.nan,
            "entropy": np.nan, "gini": np.nan,
            "burst": np.nan,
            "d_mean": np.nan, "d_std": np.nan
        }
    d = np.diff(x)
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-9)
    z = (x - m) / s
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "iqr": float(iqr(x)),
        "mad": float(mad(x)),
        "skew": skew,
        "kurt": kurt,
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "entropy": float(entropy_hist(x)),
        "gini": float(gini(x)),
        "burst": float(robust_burst_rate(x)),
        "d_mean": float(np.mean(d)),
        "d_std": float(np.std(d)),
    }

def build_mask_from_mat(mat, exclude_singing=True):
    def get(k):
        if k not in mat:
            return None
        v = np.asarray(mat[k]).squeeze()
        if v.size == 0:
            return None
        return v.reshape(-1)

    rec = get("recordingOn")
    voiced = get("voiced")
    sing = get("voiced_singing") if exclude_singing else None

    n = None
    for v in [rec, voiced, sing]:
        if v is not None and v.size > 0:
            n = v.size if n is None else min(n, v.size)
    if n is None or n == 0:
        return None

    mask = np.ones(n, dtype=bool)
    if rec is not None:
        mask &= (rec[:n] > 0)
    if voiced is not None:
        mask &= (voiced[:n] > 0)
    if exclude_singing and sing is not None:
        mask &= (sing[:n] <= 0)
    return mask

def day_features_from_mat(mat_path, exclude_singing=True, min_voiced=20):
    mat = loadmat(mat_path)
    mask = build_mask_from_mat(mat, exclude_singing=exclude_singing)

    feat = {}
    flags = {}

    if mask is None or int(mask.sum()) < min_voiced:
        dummy = stat_pack(np.array([0, 1, 2, 3, 4], dtype=np.float32))
        for k in PRIMARY_KEYS:
            for sname in dummy.keys():
                feat[f"{k}__{sname}"] = np.nan
        flags["voiced_frames"] = 0
        for ib in ["IBIF_hrf", "IBIF_h1h2", "IBIF_mfdr", "IBIF_acflow", "IBIF_sq", "IBIF_oq", "IBIF_naq", "IBIF_cq"]:
            flags[f"{ib}_missing"] = 1
        return feat, flags

    flags["voiced_frames"] = int(mask.sum())

    for k in PRIMARY_KEYS:
        if k not in mat:
            dummy = stat_pack(np.array([0, 1, 2, 3, 4], dtype=np.float32))
            for sname in dummy.keys():
                feat[f"{k}__{sname}"] = np.nan
            if k.startswith("IBIF_"):
                flags[f"{k}_missing"] = 1
            continue

        x = np.asarray(mat[k]).squeeze().reshape(-1)
        n = min(x.size, mask.size)
        x = x[:n].astype(np.float32)
        m2 = mask[:n]
        x_use = _safe_float(x[m2])

        pack = stat_pack(x_use)
        for sname, sval in pack.items():
            feat[f"{k}__{sname}"] = float(sval) if np.isfinite(sval) else np.nan

        if k.startswith("IBIF_"):
            flags[f"{k}_missing"] = 0

    return feat, flags

def aggregate_subject(day_df):
    non_feat = {"subjectID", "day_id", "mat_path"}
    feat_cols = [c for c in day_df.columns if c not in non_feat and not c.endswith("_missing") and c != "voiced_frames"]

    agg_funcs = ["mean", "median", "std", "min", "max"]
    g = day_df.groupby("subjectID", sort=False)

    blocks = []
    for fn in agg_funcs:
        tmp = g[feat_cols].agg(fn)
        tmp.columns = [f"{c}__day_{fn}" for c in tmp.columns]
        blocks.append(tmp)

    vf = g["voiced_frames"].agg(["mean", "min", "max"]).rename(
        columns={"mean": "voiced_frames__mean", "min": "voiced_frames__min", "max": "voiced_frames__max"}
    )
    blocks.append(vf)

    miss_cols = [c for c in day_df.columns if c.endswith("_missing")]
    if miss_cols:
        miss_rate = g[miss_cols].mean()
        miss_rate.columns = [f"{c}__rate" for c in miss_rate.columns]
        blocks.append(miss_rate)

    subj = pd.concat(blocks, axis=1).reset_index()
    return subj

def add_npvh_interactions(df):
    df = df.copy()
    if "cppall__median__day_mean" in df.columns and "fo__median__day_mean" in df.columns:
        df["phonation_efficiency"] = df["cppall__median__day_mean"] / (df["fo__median__day_mean"] + 1e-6)
    if "H1H2all__median__day_mean" in df.columns and "cppall__median__day_mean" in df.columns:
        df["closure_periodicity_ratio"] = df["H1H2all__median__day_mean"] / (df["cppall__median__day_mean"] + 1e-6)
    if "LHratioall__std__day_mean" in df.columns and "LHratioall__median__day_mean" in df.columns:
        df["lh_stability"] = df["LHratioall__std__day_mean"] / (df["LHratioall__median__day_mean"] + 1e-6)
    if "spectralTiltall__std__day_mean" in df.columns and "spectralTiltall__median__day_mean" in df.columns:
        df["tilt_stability"] = df["spectralTiltall__std__day_mean"] / (df["spectralTiltall__median__day_mean"].abs() + 1e-6)
    if "cppall__median__day_std" in df.columns and "cppall__median__day_mean" in df.columns:
        df["cpp_day_cv"] = df["cppall__median__day_std"] / (df["cppall__median__day_mean"] + 1e-6)
    return df

def build_subject_feature_table(subject_to_paths, exclude_singing=True, add_interactions=False):
    rows = []
    did = 0
    for sid, paths in subject_to_paths.items():
        for mp in paths:
            feat, flags = day_features_from_mat(mp, exclude_singing=exclude_singing)
            row = {"subjectID": sid, "day_id": did, "mat_path": mp}
            row.update(feat)
            row.update(flags)
            rows.append(row)
            did += 1

    day_df = pd.DataFrame(rows)
    subj = aggregate_subject(day_df)
    if add_interactions:
        subj = add_npvh_interactions(subj)
    return subj

def tab_predict_catboost(subj_df, feat_list_path, model_path,
                         do_impute_median_from_train=False,
                         train_csv="/workspace/NeckVibeChallenge/Labels/Train.csv",
                         train_feature_dir="/workspace/NeckVibeChallenge/Features",
                         add_interactions=False):

    feats = safe_read_json(feat_list_path)
    for c in feats:
        if c not in subj_df.columns:
            subj_df[c] = np.nan

    X = subj_df[feats].replace([np.inf, -np.inf], np.nan)

    if do_impute_median_from_train:
        meta = pd.read_csv(train_csv)

        mat_col = None
        for c in ["Feature filename", "feature filename", "feat_path", "mat_path", "file", "filename"]:
            if c in meta.columns:
                mat_col = c
                break
        if mat_col is None:
            for c in meta.columns:
                cl = c.lower()
                if "filename" in cl or "mat" in cl or "file" in cl or "feature" in cl:
                    mat_col = c
                    break
        if mat_col is None:
            raise ValueError("Cannot find mat filename column in Train.csv for median reconstruction.")

        train_subject_to_paths = {}
        for _, r in meta.iterrows():
            sid = str(r["subjectID"]).strip()
            mp = str(r[mat_col]).strip()
            if mp.lower() in ["nan", "none", ""]:
                continue
            mat_path = mp if os.path.isabs(mp) else os.path.join(train_feature_dir, mp)
            if not os.path.exists(mat_path):
                continue
            train_subject_to_paths.setdefault(sid, []).append(mat_path)

        train_subj = build_subject_feature_table(train_subject_to_paths, exclude_singing=True, add_interactions=add_interactions)
        for c in feats:
            if c not in train_subj.columns:
                train_subj[c] = np.nan
        Xtr = train_subj[feats].replace([np.inf, -np.inf], np.nan)
        med = Xtr.median(numeric_only=True)
        X = X.fillna(med).fillna(0)

    model = CatBoostClassifier()
    model.load_model(model_path)
    p = model.predict_proba(X)[:, 1]
    return clip01(p)


# =======================
# EXACT MIL INFERENCE
# =======================
def _load_py_module(module_name: str, py_path: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@torch.no_grad()
def mil_predict_average_folds_exact(subject_to_paths, ckpt_dir, ckpt_pattern, mil_py_path, device="cuda"):
    mod = _load_py_module(f"mil_mod_{os.path.basename(mil_py_path).replace('.','_')}", mil_py_path)

    subject_ids = sorted(list(subject_to_paths.keys()))
    subject_to_label = {sid: 0 for sid in subject_ids}  # dummy labels

    # Choose dataset class available in that file
    if hasattr(mod, "EnhancedSubjectBagDataset"):
        ds = mod.EnhancedSubjectBagDataset(
            subject_to_paths=subject_to_paths,
            subject_to_label=subject_to_label,
            subject_ids=subject_ids,
            exclude_singing=getattr(mod, "EXCLUDE_SINGING", True),
            win_frames=getattr(mod, "WIN_FRAMES", 240),
            max_windows=getattr(mod, "MAX_WINDOWS_PER_SUBJECT", 24),
        )
    elif hasattr(mod, "SubjectBagDataset"):
        ds = mod.SubjectBagDataset(
            subject_to_paths=subject_to_paths,
            subject_to_label=subject_to_label,
            subject_ids=subject_ids,
            exclude_singing=getattr(mod, "EXCLUDE_SINGING", True),
            win_frames=getattr(mod, "WIN_FRAMES", 240),
            max_windows=getattr(mod, "MAX_WINDOWS_PER_SUBJECT", 24),
        )
    else:
        raise RuntimeError(f"Cannot find EnhancedSubjectBagDataset or SubjectBagDataset in {mil_py_path}")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # infer channels from dataset output
    sample_bag = ds[0][0]  # (K, C, T)
    in_ch = int(sample_bag.shape[1])

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, ckpt_pattern)))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found: {os.path.join(ckpt_dir, ckpt_pattern)}")

    fold_preds = []

    for ckpt in ckpts:
        model = mod.MILClassifier(in_ch=in_ch).to(device)

        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]

        model.load_state_dict(state, strict=False)
        model.eval()

        preds = {}
        for batch in loader:
            bag = batch[0].to(device, non_blocking=True)
            sids = batch[-1]

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logit, _ = model(bag)
                p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)

            for s, pv in zip(sids, p):
                preds[str(s)] = float(pv)

        fold_preds.append(preds)

    out = {}
    for sid in subject_ids:
        vals = [d.get(sid, 0.5) for d in fold_preds]
        out[sid] = float(np.mean(vals))
    return out


# =======================
# STACKING APPLY
# =======================
def make_meta_features(t, m):
    t = np.asarray(t).reshape(-1)
    m = np.asarray(m).reshape(-1)
    return np.stack([t, m, np.abs(t - m), t * m], axis=1)

def apply_stacking(meta, qt_tab, qt_mil, tab_pred, mil_pred):
    tab_pred = np.asarray(tab_pred, dtype=float).reshape(-1, 1)
    mil_pred = np.asarray(mil_pred, dtype=float).reshape(-1, 1)
    t = qt_tab.transform(tab_pred).reshape(-1)
    m = qt_mil.transform(mil_pred).reshape(-1)
    X = make_meta_features(t, m)
    p = meta.predict_proba(X)[:, 1]
    return clip01(p)


# =======================
# OUTPUT 
# =======================
def populate_task(template_csv, out_csv, prob_col, pred_col, probs_by_sid):
    df = pd.read_csv(template_csv)
    if "subjectID" not in df.columns:
        raise ValueError(f"Template missing subjectID column: {template_csv}")

    probs = []
    labels = []
    for sid in df["subjectID"].astype(str).tolist():
        pv = float(probs_by_sid.get(sid, 0.5))
        pv = float(np.clip(pv, 0.0, 1.0))
        probs.append(pv)
        labels.append(int(pv >= THRESH))

    df[prob_col] = probs
    df[pred_col] = labels
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} (rows={len(df)})")


# =======================
# MAIN
# =======================
def main():
    subject_to_paths = build_subject_to_paths(TEST_FEATURE_DIR, TEST_INDEX_CSV)
    if len(subject_to_paths) == 0:
        raise RuntimeError(f"No test .mat files found under: {TEST_FEATURE_DIR}")

    print(f"[TEST] Subjects found: {len(subject_to_paths)}")

    sids = sorted(subject_to_paths.keys())

    # -------- PVH --------
    print("\n[PVH] Tabular inference...")
    subj_pvh = build_subject_feature_table(subject_to_paths, exclude_singing=True, add_interactions=False)
    pvh_tab = tab_predict_catboost(subj_pvh, PVH_TAB_FEATS, PVH_TAB_MODEL, do_impute_median_from_train=False)
    pvh_tab_by_sid = dict(zip(subj_pvh["subjectID"].astype(str).tolist(), pvh_tab.tolist()))

    print("[PVH] MIL inference (exact script, averaging folds)...")
    pvh_mil_by_sid = mil_predict_average_folds_exact(
        subject_to_paths=subject_to_paths,
        ckpt_dir=PVH_MIL_DIR,
        ckpt_pattern="mil_pvh_fold*.pt",
        mil_py_path=PVH_MIL_PY,
        device=DEVICE
    )

    pvh_meta = load_pickle(PVH_META_PKL)
    pvh_qt_tab = load_pickle(PVH_QT_TAB_PKL)
    pvh_qt_mil = load_pickle(PVH_QT_MIL_PKL)

    pvh_tab_vec = np.array([pvh_tab_by_sid.get(s, 0.5) for s in sids], dtype=float)
    pvh_mil_vec = np.array([pvh_mil_by_sid.get(s, 0.5) for s in sids], dtype=float)

    pvh_ens = apply_stacking(pvh_meta, pvh_qt_tab, pvh_qt_mil, pvh_tab_vec, pvh_mil_vec)
    pvh_ens_by_sid = {sid: float(p) for sid, p in zip(sids, pvh_ens)}

    # -------- NPVH --------
    print("\n[NPVH] Tabular inference...")
    subj_npvh = build_subject_feature_table(subject_to_paths, exclude_singing=True, add_interactions=True)
    npvh_tab = tab_predict_catboost(
        subj_npvh, NPVH_TAB_FEATS, NPVH_TAB_MODEL,
        do_impute_median_from_train=True,
        add_interactions=True
    )
    npvh_tab_by_sid = dict(zip(subj_npvh["subjectID"].astype(str).tolist(), npvh_tab.tolist()))

    print("[NPVH] MIL inference (exact script, averaging folds)...")
    npvh_mil_by_sid = mil_predict_average_folds_exact(
        subject_to_paths=subject_to_paths,
        ckpt_dir=NPVH_MIL_DIR,
        ckpt_pattern="mil_npvh_fold*.pt",
        mil_py_path=NPVH_MIL_PY,
        device=DEVICE
    )

    npvh_meta = load_pickle(NPVH_META_PKL)
    npvh_qt_tab = load_pickle(NPVH_QT_TAB_PKL)
    npvh_qt_mil = load_pickle(NPVH_QT_MIL_PKL)

    npvh_tab_vec = np.array([npvh_tab_by_sid.get(s, 0.5) for s in sids], dtype=float)
    npvh_mil_vec = np.array([npvh_mil_by_sid.get(s, 0.5) for s in sids], dtype=float)

    npvh_ens = apply_stacking(npvh_meta, npvh_qt_tab, npvh_qt_mil, npvh_tab_vec, npvh_mil_vec)
    npvh_ens_by_sid = {sid: float(p) for sid, p in zip(sids, npvh_ens)}

    # -------- Populate templates --------
    out_pvh = os.path.join(OUT_DIR, "Task1_PVH_detection.csv")
    out_npvh = os.path.join(OUT_DIR, "Task2_NPVH_detection.csv")

    populate_task(TEMPLATE_PVH, out_pvh, "pvh_probability", "pvh_prediction", pvh_ens_by_sid)
    populate_task(TEMPLATE_NPVH, out_npvh, "npvh_probability", "npvh_prediction", npvh_ens_by_sid)

    print("\nDone.")
    print("Submission files created in:", OUT_DIR)


if __name__ == "__main__":
    main()
