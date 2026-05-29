import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG
# =========================
TRAIN_CSV = "/workspace/NeckVibeChallenge/Labels/Train.csv"
FEATURE_DIR = "/workspace/NeckVibeChallenge/Features"

USE_CATBOOST = True
EXCLUDE_SINGING = True
FOLDS = 5
SEED = 42
OUT_DIR = "./npvh_out_final"
# =========================

PRIMARY_KEYS = [
     "H1H2all", "LHratioall", "cppall", "dBcms2", "spectralTiltall", "level",
    "IBIF_h1h2", "IBIF_hrf", "IBIF_mfdr", "IBIF_acflow", "IBIF_sq",
    "IBIF_oq", "IBIF_naq", "IBIF_cq",]


# -------------------------
# Helpers
# -------------------------
def _safe(x):
    x = np.asarray(x, dtype=np.float32).squeeze()
    x = x[np.isfinite(x)]
    return x

def mad(x):
    x = _safe(x)
    if x.size == 0: return np.nan
    m = np.median(x)
    return float(np.median(np.abs(x - m)))

def iqr(x):
    x = _safe(x)
    if x.size == 0: return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def entropy_hist(x, bins=32):
    x = _safe(x)
    if x.size == 0: return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0: return 0.0
    p = hist / np.sum(hist)
    return float(-np.sum(p * np.log(p + 1e-12)))

def stat_pack(x):
    x = _safe(x)
    if x.size < 5:
        return {
            "mean": np.nan, "std": np.nan, "median": np.nan,
            "iqr": np.nan, "mad": np.nan,
            "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan,
            "entropy": np.nan,
            "skew": np.nan, "kurt": np.nan, "range": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "iqr": iqr(x),
        "mad": mad(x),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "entropy": entropy_hist(x),
        "skew": float(stats.skew(x)) if x.size >= 3 else np.nan,
        "kurt": float(stats.kurtosis(x)) if x.size >= 4 else np.nan,
        "range": float(np.ptp(x)),
    }

def build_mask(mat, exclude_singing=True):
    def get(k):
        if k not in mat: return None
        v = np.asarray(mat[k]).squeeze()
        return v if v.size > 0 else None

    rec = get("recordingOn")
    voiced = get("voiced")
    sing = get("voiced_singing") if exclude_singing else None

    n = None
    for v in [rec, voiced, sing]:
        if v is not None:
            n = v.size
            break
    if n is None or n == 0:
        return None

    mask = np.ones(n, dtype=bool)
    if rec is not None: mask &= (rec > 0)
    if voiced is not None: mask &= (voiced > 0)
    if exclude_singing and sing is not None: mask &= (sing <= 0)
    return mask

def rank_normalize(p):
    """Fold-wise rank normalization to make scores comparable across folds."""
    p = np.asarray(p, dtype=np.float32)
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = (np.arange(len(p), dtype=np.float32) + 1) / (len(p) + 1)
    return out

# -------------------------
# Advanced NPVH features
# -------------------------
def npvh_cpp_features(cpp_vals):
    x = _safe(cpp_vals)
    if x.size < 20:
        return {
            "cpp_low_rate": np.nan,
            "cpp_cv": np.nan,
            "cpp_p05_minus_med": np.nan,
            "cpp_below_14": np.nan,
            "cpp_range": np.nan,
            "cpp_kurt": np.nan,
            "cpp_skew": np.nan,
        }
    med = np.median(x)
    low_thr = np.percentile(x, 15)
    low_rate = float(np.mean(x <= low_thr))
    sd = np.std(x) + 1e-6
    cv = float(sd / (np.mean(x) + 1e-6))
    return {
        "cpp_low_rate": low_rate,
        "cpp_cv": cv,
        "cpp_p05_minus_med": float(np.percentile(x, 5) - med),
        "cpp_below_14": float(np.mean(x < 14.45)),
        "cpp_range": float(np.ptp(x)),
        "cpp_kurt": float(stats.kurtosis(x)) if x.size >= 4 else np.nan,
        "cpp_skew": float(stats.skew(x)) if x.size >= 3 else np.nan,
    }

def npvh_advanced_features(mat, mask):
    features = {}

    cpp = _safe(np.asarray(mat.get("cppall", [])).squeeze())
    if mask is not None and cpp.size == mask.size:
        cpp = cpp[mask]
    if cpp.size >= 20:
        features.update(npvh_cpp_features(cpp))

    fo = _safe(np.asarray(mat.get("fo", [])).squeeze())
    if mask is not None and fo.size == mask.size:
        fo = fo[mask]
    if cpp.size >= 20 and fo.size >= 20:
        features["fo_cpp_ratio"] = float(np.mean(fo) / (np.mean(cpp) + 1e-6))
        mlen = min(len(fo), len(cpp))
        features["fo_cpp_corr"] = float(np.corrcoef(fo[:mlen], cpp[:mlen])[0, 1]) if mlen >= 3 else np.nan

    h1h2 = _safe(np.asarray(mat.get("H1H2all", [])).squeeze())
    if mask is not None and h1h2.size == mask.size:
        h1h2 = h1h2[mask]
    if h1h2.size >= 20:
        features["h1h2_skew"] = float(stats.skew(h1h2)) if h1h2.size >= 3 else np.nan
        features["h1h2_extreme_rate"] = float(np.mean(np.abs(h1h2 - np.median(h1h2)) > 2*np.std(h1h2)))
        if cpp.size >= 20:
            features["h1h2_cpp_ratio"] = float(np.mean(h1h2) / (np.mean(cpp) + 1e-6))

    tilt = _safe(np.asarray(mat.get("spectralTiltall", [])).squeeze())
    if mask is not None and tilt.size == mask.size:
        tilt = tilt[mask]
    if tilt.size >= 20:
        features["tilt_extreme_rate"] = float(np.mean(np.abs(tilt - np.median(tilt)) > 2*np.std(tilt)))
        features["tilt_range"] = float(np.ptp(tilt))

    lh = _safe(np.asarray(mat.get("LHratioall", [])).squeeze())
    if mask is not None and lh.size == mask.size:
        lh = lh[mask]
    if lh.size >= 20:
        features["lh_cv"] = float(np.std(lh) / (np.mean(lh) + 1e-6))
        features["lh_range"] = float(np.ptp(lh))

    level = _safe(np.asarray(mat.get("level", [])).squeeze())
    if mask is not None and level.size == mask.size:
        level = level[mask]
    if level.size >= 20:
        features["level_cv"] = float(np.std(level) / (np.mean(level) + 1e-6))

    return features

# -------------------------
# Day feature extraction
# -------------------------
def day_features_from_mat(mat_path, exclude_singing=True):
    mat = loadmat(mat_path)
    mask = build_mask(mat, exclude_singing=exclude_singing)

    feat = {}
    flags = {}

    if mask is None or mask.sum() < 20:
        for k in PRIMARY_KEYS:
            pack = stat_pack(np.array([0,1,2,3,4], dtype=np.float32))
            for sname in pack.keys():
                feat[f"{k}__{sname}"] = np.nan

        adv = npvh_advanced_features({}, None)
        feat.update({k: np.nan for k in adv.keys()})

        flags["voiced_frames"] = 0
        for ib in ["IBIF_hrf", "IBIF_h1h2", "IBIF_mfdr"]:
            flags[f"{ib}_missing"] = 1
        return feat, flags

    flags["voiced_frames"] = int(mask.sum())

    for k in PRIMARY_KEYS:
        if k not in mat:
            pack = stat_pack(np.array([0,1,2,3,4], dtype=np.float32))
            for sname in pack.keys():
                feat[f"{k}__{sname}"] = np.nan
            if k.startswith("IBIF_"):
                flags[f"{k}_missing"] = 1
            continue

        x = np.asarray(mat[k]).squeeze()
        x_use = x[mask] if x.size == mask.size else x

        pack = stat_pack(x_use)
        for sname, sval in pack.items():
            feat[f"{k}__{sname}"] = sval

        if k.startswith("IBIF_"):
            flags[f"{k}_missing"] = 0

    feat.update(npvh_advanced_features(mat, mask))
    return feat, flags

# -------------------------
# Subject aggregation
# -------------------------
def add_subject_trend_features(day_df):
    out_rows = []
    for sid, g in day_df.groupby("subjectID", sort=False):
        g = g.sort_values("day_id")
        t = np.arange(len(g), dtype=np.float32)
        row = {"subjectID": sid}

        for col in ["cppall__median", "H1H2all__median", "spectralTiltall__median",
                    "LHratioall__median", "fo__median", "level__median"]:
            if col not in g.columns or len(g) < 3:
                row[f"{col}__slope"] = np.nan
                continue
            y = g[col].astype(float).values
            m = np.isfinite(y)
            if m.sum() < 3:
                row[f"{col}__slope"] = np.nan
                continue
            tt = t[m]
            yy = y[m]
            A = np.vstack([tt, np.ones_like(tt)]).T
            row[f"{col}__slope"] = float(np.linalg.lstsq(A, yy, rcond=None)[0][0])

        out_rows.append(row)
    return pd.DataFrame(out_rows)

def aggregate_subject(day_df):
    non_feat = {"subjectID", "day_id", "mat_path", "y_day", "subject_y"}
    feat_cols = [c for c in day_df.columns if c not in non_feat and not c.endswith("_missing") and c != "voiced_frames"]

    g = day_df.groupby("subjectID", sort=False)
    blocks = []

    for fn in ["mean", "median", "std", "min", "max"]:
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
    subj = subj.merge(add_subject_trend_features(day_df), on="subjectID", how="left")
    return subj

def add_interaction_features(df):
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

# -------------------------
# Model
# -------------------------
def try_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None

def build_model(use_catboost=True, seed=42):
    CatBoost = try_catboost()
    if use_catboost and CatBoost is not None:
        model = CatBoost(
            iterations=6000,
            learning_rate=0.015,
            depth=5,
            l2_leaf_reg=15.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=200,
            od_type="Iter",
            od_wait=400,
            border_count=64,
            min_data_in_leaf=3,
            allow_writing_files=False,
        )
        return ("catboost", model)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=0.5,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed
        ))
    ])
    return ("logreg", lr)

def simple_impute(X_train, X_val):
    med = X_train.median()
    return X_train.fillna(med).fillna(0), X_val.fillna(med).fillna(0)

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = pd.read_csv(TRAIN_CSV)

    if "subjectID" not in meta.columns:
        raise ValueError("Train.csv must contain subjectID")
    if "groupLabel" not in meta.columns:
        raise ValueError("Train.csv must contain groupLabel")

    meta["subjectID"] = meta["subjectID"].astype(str).str.strip()

    # NPVH (1) vs non-NPVH (0)
    y_day = (meta["groupLabel"].astype(str) == "NPVH").astype(int).values

    # filename column
    mat_col = None
    preferred = ["Feature filename", "feature filename", "feat_path", "mat_path", "file", "filename"]
    for c in preferred:
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
        raise ValueError(f"Cannot find filename column. Columns: {list(meta.columns)}")

    # build day_df
    day_rows = []
    missing = 0
    for i, r in meta.iterrows():
        sid = str(r["subjectID"]).strip()
        mp = str(r[mat_col]).strip()
        if mp.lower() in ["nan", "none", ""]:
            continue
        mat_path = mp if os.path.isabs(mp) else os.path.join(FEATURE_DIR, mp)
        if not os.path.exists(mat_path):
            missing += 1
            continue

        feat, flags = day_features_from_mat(mat_path, exclude_singing=EXCLUDE_SINGING)
        row = {"subjectID": sid, "day_id": i, "mat_path": mat_path, "y_day": int(y_day[i])}
        row.update(feat)
        row.update(flags)
        day_rows.append(row)

    day_df = pd.DataFrame(day_rows)
    if day_df.empty:
        raise RuntimeError("No day rows built. Check FEATURE_DIR/path column.")

    day_df["subject_y"] = day_df.groupby("subjectID")["y_day"].transform("max")

    print(f"[NPVH-FINAL] Total days: {len(day_df)}, Missing mats: {missing}")
    print(f"[NPVH-FINAL] Total subjects: {day_df['subjectID'].nunique()}")
    print(f"[NPVH-FINAL] Class balance: {day_df.groupby('subject_y')['subjectID'].nunique().to_dict()}")

    unique_subjects = day_df[["subjectID", "subject_y"]].drop_duplicates().reset_index(drop=True)
    subject_groups = unique_subjects["subjectID"].values
    subject_y = unique_subjects["subject_y"].values.astype(int)

    # GLOBAL class weight (stable across folds)
    global_pos = int(subject_y.sum())
    global_neg = int((subject_y == 0).sum())
    GLOBAL_SPW = global_neg / max(1, global_pos)
    print(f"[NPVH-FINAL] GLOBAL scale_pos_weight = {GLOBAL_SPW:.4f}")

    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    # OOF: raw + rank-normalized (comparable across folds)
    oof_raw = np.full(len(unique_subjects), np.nan, dtype=np.float32)
    oof_rank = np.full(len(unique_subjects), np.nan, dtype=np.float32)

    fold_aucs = []
    fold_sizes = []
    fi_list = []

    model_name, _ = build_model(use_catboost=USE_CATBOOST, seed=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(unique_subjects, subject_y, subject_groups), 1):
        tr_subjects = set(unique_subjects.iloc[tr_idx]["subjectID"].astype(str).values)
        va_subjects = set(unique_subjects.iloc[va_idx]["subjectID"].astype(str).values)

        day_train = day_df[day_df["subjectID"].isin(tr_subjects)].copy()
        day_val = day_df[day_df["subjectID"].isin(va_subjects)].copy()

        subj_tr = add_interaction_features(aggregate_subject(day_train))
        subj_va = add_interaction_features(aggregate_subject(day_val))

        y_tr_map = day_train.groupby("subjectID")["y_day"].max()
        y_va_map = day_val.groupby("subjectID")["y_day"].max()

        subj_tr = subj_tr.merge(y_tr_map.rename("y").reset_index(), on="subjectID", how="inner")
        subj_va = subj_va.merge(y_va_map.rename("y").reset_index(), on="subjectID", how="inner")

        subj_tr["subjectID"] = subj_tr["subjectID"].astype(str)
        subj_va["subjectID"] = subj_va["subjectID"].astype(str)

        feat_cols = [c for c in subj_tr.columns if c not in ["subjectID", "y"]]
        Xtr = subj_tr[feat_cols].replace([np.inf, -np.inf], np.nan)
        Xva = subj_va[feat_cols].replace([np.inf, -np.inf], np.nan)

        ytr = subj_tr["y"].values.astype(int)
        yva = subj_va["y"].values.astype(int)

        Xtr_f, Xva_f = simple_impute(Xtr, Xva)

        if model_name == "catboost":
            m = build_model(use_catboost=True, seed=SEED)[1]
            m.set_params(scale_pos_weight=GLOBAL_SPW)
            m.fit(Xtr_f, ytr, eval_set=(Xva_f, yva), use_best_model=True)
            p = m.predict_proba(Xva_f)[:, 1].astype(np.float32)

            fi_list.append(pd.DataFrame({
                "feature": feat_cols,
                "importance": m.feature_importances_,
                "fold": fold
            }))
        else:
            m = build_model(use_catboost=False, seed=SEED)[1]
            m.fit(Xtr_f, ytr)
            p = m.predict_proba(Xva_f)[:, 1].astype(np.float32)

        # fold AUC (raw)
        auc = roc_auc_score(yva, p)
        fold_aucs.append(float(auc))
        fold_sizes.append(len(yva))

        # rank-normalized within fold (for pooled OOF comparability)
        p_r = rank_normalize(p)

        # Fill OOF by indices (guaranteed alignment)
        filled = 0
        va_sids = unique_subjects.iloc[va_idx]["subjectID"].astype(str).values
        sid2p = dict(zip(subj_va["subjectID"].values, p))
        sid2pr = dict(zip(subj_va["subjectID"].values, p_r))

        for j, sid in zip(va_idx, va_sids):
            oof_raw[j] = float(sid2p[sid])
            oof_rank[j] = float(sid2pr[sid])
            filled += 1

        print(f"[Fold {fold}] Filled OOF: {filled}/{len(va_idx)} | Fold AUC(raw)={auc:.4f} | Pos={yva.sum()}")

    # hard checks
    if np.isnan(oof_raw).any() or np.isnan(oof_rank).any():
        raise RuntimeError("OOF contains NaNs. This should never happen now.")

    # metrics
    mean_fold = float(np.mean(fold_aucs))
    std_fold = float(np.std(fold_aucs))
    weighted_fold = float(np.average(fold_aucs, weights=np.array(fold_sizes, dtype=np.float32)))

    pooled_raw = float(roc_auc_score(subject_y, oof_raw))
    pooled_rank = float(roc_auc_score(subject_y, oof_rank))

    print("\n[NPVH-FINAL] ===== FINAL CV RESULTS =====")
    print(f"[NPVH-FINAL] Fold AUCs (raw): {np.round(fold_aucs, 4).tolist()}")
    print(f"[NPVH-FINAL] Mean Fold AUC (raw): {mean_fold:.4f} (+/- {std_fold:.4f})")
    print(f"[NPVH-FINAL] Weighted Mean Fold AUC (raw): {weighted_fold:.4f}")
    print(f"[NPVH-FINAL] Pooled OOF AUC (raw): {pooled_raw:.4f}  <-- can be lower due to fold scale drift")
    print(f"[NPVH-FINAL] Pooled OOF AUC (rank-normalized): {pooled_rank:.4f}  <-- comparable across folds")

    # -------------------------
    # Train final model on all subjects
    # -------------------------
    subj_full = add_interaction_features(aggregate_subject(day_df))
    y_full = day_df.groupby("subjectID")["y_day"].max().rename("y").reset_index()
    subj_full = subj_full.merge(y_full, on="subjectID", how="inner")
    subj_full["subjectID"] = subj_full["subjectID"].astype(str)

    feat_cols_final = [c for c in subj_full.columns if c not in ["subjectID", "y"]]
    X_full = subj_full[feat_cols_final].replace([np.inf, -np.inf], np.nan)
    y_full = subj_full["y"].values.astype(int)

    med_full = X_full.median()
    X_full_imp = X_full.fillna(med_full).fillna(0)

    final_model_name, _ = build_model(use_catboost=USE_CATBOOST, seed=SEED)

    if final_model_name == "catboost":
        from catboost import CatBoostClassifier  # safe now
        final_model = build_model(use_catboost=True, seed=SEED)[1]
        final_model.set_params(scale_pos_weight=GLOBAL_SPW)
        final_model.fit(X_full_imp, y_full, verbose=200)
        model_path = os.path.join(OUT_DIR, "npvh_catboost_final.cbm")
        final_model.save_model(model_path)
    else:
        import joblib
        final_model = build_model(use_catboost=False, seed=SEED)[1]
        final_model.fit(X_full_imp, y_full)
        model_path = os.path.join(OUT_DIR, "npvh_logreg_final.joblib")
        joblib.dump({"model": final_model, "median": med_full}, model_path)

    # save artifacts
    with open(os.path.join(OUT_DIR, "npvh_features_final.json"), "w") as f:
        json.dump(feat_cols_final, f, indent=2)

    pd.DataFrame({
        "subjectID": unique_subjects["subjectID"].astype(str).values,
        "y": subject_y.astype(int),
        "oof_raw": oof_raw.astype(float),
        "oof_rank": oof_rank.astype(float),
    }).to_csv(os.path.join(OUT_DIR, "npvh_oof_final.csv"), index=False)

    if fi_list:
        fi = pd.concat(fi_list, ignore_index=True)
        fi_mean = fi.groupby("feature")["importance"].mean().sort_values(ascending=False)
        fi_mean.to_csv(os.path.join(OUT_DIR, "feature_importance_final.csv"))
        print("\n[NPVH-FINAL] Top 20 features:")
        print(fi_mean.head(20))

    print(f"\n[NPVH-FINAL] Saved model: {model_path}")
    print(f"[NPVH-FINAL] Outputs in: {OUT_DIR}")
    print(f"[NPVH-FINAL] Total features: {len(feat_cols_final)}")


if __name__ == "__main__":
    main()
