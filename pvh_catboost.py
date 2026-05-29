import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================
# CONFIG 
# =========================
TRAIN_CSV = "/workspace/NeckVibeChallenge/Labels/Train.csv"
FEATURE_DIR = "/workspace/NeckVibeChallenge/Features"

USE_CATBOOST = True          
EXCLUDE_SINGING = True       
FOLDS = 5                    
SEED = 42
OUT_DIR = "./pvh_out"        



# --------------------------
# Utility
# --------------------------

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
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "iqr": float(iqr(x)),
        "mad": float(mad(x)),
        "skew": float(skew(x)),
        "kurt": float(kurtosis(x)),
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


PRIMARY_KEYS = [
    "H1H2all", "LHratioall", "cppall", "dBcms2", "spectralTiltall", "level",
    "IBIF_h1h2", "IBIF_hrf", "IBIF_mfdr", "IBIF_acflow", "IBIF_sq",
    "IBIF_oq", "IBIF_naq", "IBIF_cq",]

# --------------------------
# Mask + daily features
# --------------------------

def build_mask(mat_path, exclude_singing=True):
    mat = loadmat(mat_path)

    def get(k):
        if k not in mat:
            return None
        return np.asarray(mat[k]).squeeze()

    rec = get("recordingOn")
    voiced = get("voiced")
    sing = get("voiced_singing") if exclude_singing else None

    n = None
    for v in [rec, voiced, sing]:
        if v is not None and np.asarray(v).size > 0:
            n = np.asarray(v).size
            break
    if n is None or n == 0:
        return None

    mask = np.ones(n, dtype=bool)
    if rec is not None:
        mask &= (np.asarray(rec).squeeze() > 0)
    if voiced is not None:
        mask &= (np.asarray(voiced).squeeze() > 0)
    if exclude_singing and sing is not None:
        mask &= (np.asarray(sing).squeeze() <= 0)

    return mask


def day_features_from_mat(mat_path, exclude_singing=True):
    mask = build_mask(mat_path, exclude_singing=exclude_singing)

    feat = {}
    flags = {}

    if mask is None or mask.sum() < 20:
        dummy_pack = stat_pack(np.array([0, 1, 2, 3, 4], dtype=np.float32))
        for k in PRIMARY_KEYS:
            for sname in dummy_pack.keys():
                feat[f"{k}__{sname}"] = np.nan
        flags["voiced_frames"] = 0
        for ib in ["IBIF_hrf", "IBIF_h1h2", "IBIF_mfdr"]:
            flags[f"{ib}_missing"] = 1
        return feat, flags

    flags["voiced_frames"] = int(mask.sum())

    mat = loadmat(mat_path)
    for k in PRIMARY_KEYS:
        if k not in mat:
            dummy_pack = stat_pack(np.array([0, 1, 2, 3, 4], dtype=np.float32))
            for sname in dummy_pack.keys():
                feat[f"{k}__{sname}"] = np.nan
            if k.startswith("IBIF_"):
                flags[f"{k}_missing"] = 1
            continue

        x = np.asarray(mat[k]).squeeze()
        if x.size == mask.size:
            x_use = _safe_float(x[mask])
        else:
            x_use = _safe_float(x)

        pack = stat_pack(x_use)
        for sname, sval in pack.items():
            feat[f"{k}__{sname}"] = sval

        if k.startswith("IBIF_"):
            flags[f"{k}_missing"] = 0

    return feat, flags


# --------------------------
# Day -> Subject aggregation
# --------------------------

def aggregate_subject(day_df, subject_col="subjectID"):
    non_feat = {subject_col, "day_id", "mat_path"}
    feat_cols = [c for c in day_df.columns if c not in non_feat and not c.endswith("_missing") and c != "voiced_frames"]

    agg_funcs = ["mean", "median", "std", "min", "max"]
    g = day_df.groupby(subject_col, sort=False)

    subj_blocks = []
    for fn in agg_funcs:
        tmp = g[feat_cols].agg(fn)
        tmp.columns = [f"{c}__day_{fn}" for c in tmp.columns]
        subj_blocks.append(tmp)

    vf = g["voiced_frames"].agg(["mean", "min", "max"]).rename(
        columns={"mean": "voiced_frames__mean", "min": "voiced_frames__min", "max": "voiced_frames__max"}
    )
    subj_blocks.append(vf)

    miss_cols = [c for c in day_df.columns if c.endswith("_missing")]
    if miss_cols:
        miss_rate = g[miss_cols].mean()
        miss_rate.columns = [f"{c}__rate" for c in miss_rate.columns]
        subj_blocks.append(miss_rate)

    subj = pd.concat(subj_blocks, axis=1).reset_index()
    return subj


# --------------------------
# Model
# --------------------------

def try_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None


def build_model(use_catboost=True, seed=42):
    CatBoostClassifier = try_catboost()
    if use_catboost and CatBoostClassifier is not None:
        model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=200,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False
        )
        return ("catboost", model)

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed
        ))
    ])
    return ("logreg", lr)


# --------------------------
# Main
# --------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = pd.read_csv(TRAIN_CSV)

    if "subjectID" not in meta.columns:
        raise ValueError(f"Train.csv must contain subjectID. Available columns: {list(meta.columns)}")
    if "groupLabel" not in meta.columns:
        raise ValueError(f"Train.csv must contain groupLabel. Available columns: {list(meta.columns)}")

    # PVH label
    y_day = (meta["groupLabel"].astype(str) == "PVH").astype(int).values

    
    mat_col = None
    candidates = [
        "Feature filename", "feature filename",
        "feat_path", "mat_path", "file", "filename"
    ]
    for c in candidates:
        if c in meta.columns:
            mat_col = c
            break
    if mat_col is None:
        for c in meta.columns:
            if "filename" in c.lower():
                mat_col = c
                break
    if mat_col is None:
        raise ValueError(f"Train.csv missing a filename column. Available columns: {list(meta.columns)}")

    day_rows = []
    missing_files = 0

    for i, r in meta.iterrows():
        sid = r["subjectID"]
        mp = str(r[mat_col]).strip()
        if mp.lower() in ["nan", "none", ""]:
            continue

        mat_path = mp if os.path.isabs(mp) else os.path.join(FEATURE_DIR, mp)
        if not os.path.exists(mat_path):
            missing_files += 1
            continue

        feat, flags = day_features_from_mat(mat_path, exclude_singing=EXCLUDE_SINGING)

        row = {"subjectID": sid, "day_id": i, "mat_path": mat_path, "y_day": int(y_day[i])}
        row.update(feat)
        row.update(flags)
        day_rows.append(row)

    day_df = pd.DataFrame(day_rows)
    if day_df.empty:
        raise RuntimeError(
            "No day rows built. Check FEATURE_DIR and filename column.\n"
            f"Detected filename column: {mat_col}\n"
            f"Missing files count: {missing_files}"
        )

    subj_y = day_df.groupby("subjectID")["y_day"].max().reset_index().rename(columns={"y_day": "y"})
    subj = aggregate_subject(day_df.drop(columns=["y_day"]), subject_col="subjectID").merge(
        subj_y, on="subjectID", how="inner"
    )

    feature_cols = [c for c in subj.columns if c not in ["subjectID", "y"]]
    X = subj[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = subj["y"].values.astype(int)
    groups = subj["subjectID"].values

    model_name, _ = build_model(use_catboost=USE_CATBOOST, seed=SEED)
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(subj), dtype=float)
    fold_aucs = []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        if model_name == "catboost":
            pos = max(1, int(ytr.sum()))
            neg = max(1, int((ytr == 0).sum()))
            scale_pos = neg / pos

            m = build_model(use_catboost=True, seed=SEED)[1]
            m.set_params(scale_pos_weight=scale_pos)
            m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            p = m.predict_proba(Xva)[:, 1]
        else:
            med = Xtr.median(numeric_only=True)
            m = build_model(use_catboost=False, seed=SEED)[1]
            m.fit(Xtr.fillna(med), ytr)
            p = m.predict_proba(Xva.fillna(med))[:, 1]

        oof[va] = p
        auc = roc_auc_score(yva, p)
        fold_aucs.append(auc)
        print(f"[PVH] Fold {fold}: AUC={auc:.4f}")

    overall = roc_auc_score(y, oof)
    print(f"\n[PVH] OOF AUC: {overall:.4f}")
    print(f"[PVH] Fold AUCs: {np.round(fold_aucs, 4).tolist()}  Mean={np.mean(fold_aucs):.4f}")

    # Train final model on full data
    if model_name == "catboost":
        pos = max(1, int(y.sum()))
        neg = max(1, int((y == 0).sum()))
        scale_pos = neg / pos

        final_model = build_model(use_catboost=True, seed=SEED)[1]
        final_model.set_params(scale_pos_weight=scale_pos)
        final_model.fit(X, y, verbose=200)

        model_path = os.path.join(OUT_DIR, "pvh_catboost.cbm")
        final_model.save_model(model_path)
    else:
        med = X.median(numeric_only=True)
        final_model = build_model(use_catboost=False, seed=SEED)[1]
        final_model.fit(X.fillna(med), y)

        model_path = os.path.join(OUT_DIR, "pvh_logreg.joblib")
        try:
            import joblib
            joblib.dump({"model": final_model, "median": med}, model_path)
        except Exception:
            pass

    with open(os.path.join(OUT_DIR, "pvh_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    pd.DataFrame({"subjectID": subj["subjectID"], "y": y, "oof_p": oof}).to_csv(
        os.path.join(OUT_DIR, "pvh_oof.csv"), index=False
    )

    print(f"\nDone. Model saved at: {model_path}")
    print(f"Outputs in: {OUT_DIR}")
    if missing_files > 0:
        print(f"WARNING: {missing_files} .mat files referenced in Train.csv were not found under FEATURE_DIR.")


if __name__ == "__main__":
    main()
