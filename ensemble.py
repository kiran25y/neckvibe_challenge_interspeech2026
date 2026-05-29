import os, json
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def pick_col(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing {name}. Tried {candidates}. Have {df.columns.tolist()}")


def compute_metrics(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
    }


def fit_quantile(x, n_quantiles=200):
    # map to uniform(0,1) in a stable way
    x = np.asarray(x).reshape(-1, 1)
    n_q = int(min(n_quantiles, max(10, len(x))))
    qt = QuantileTransformer(
        n_quantiles=n_q,
        output_distribution="uniform",
        subsample=int(1e9),
        random_state=42
    )
    qt.fit(x)
    return qt


def make_meta_features(t, m):
    t = np.asarray(t).reshape(-1)
    m = np.asarray(m).reshape(-1)
    X = np.stack([t, m, np.abs(t - m), t * m], axis=1)
    return X


def train_stack_oof(df, tab_col, mil_col, n_splits=5, seed=42):
    """
    Trains stacking in a CV way *on the OOF table itself* to estimate ensemble AUC.
    This is only for validation of the ensemble method (not needed for submission).
    """
    y = df["y"].astype(int).values
    tab = df[tab_col].astype(float).values
    mil = df[mil_col].astype(float).values

    # Fit QT on full OOF 
    qt_tab = fit_quantile(tab)
    qt_mil = fit_quantile(mil)

    t_all = qt_tab.transform(tab.reshape(-1, 1)).reshape(-1)
    m_all = qt_mil.transform(mil.reshape(-1, 1)).reshape(-1)

    X_all = make_meta_features(t_all, m_all)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_meta = np.zeros(len(df), dtype=float)

    for tr, va in skf.split(X_all, y):
        meta = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            max_iter=5000
        )
        meta.fit(X_all[tr], y[tr])
        oof_meta[va] = meta.predict_proba(X_all[va])[:, 1]

    return oof_meta, qt_tab, qt_mil


def train_final_meta(df, tab_col, mil_col, seed=42):
    """
    Train the final meta model on ALL OOF rows (this is what you use for submission),
    plus the quantile transformers learned on OOF.
    """
    y = df["y"].astype(int).values
    tab = df[tab_col].astype(float).values
    mil = df[mil_col].astype(float).values

    qt_tab = fit_quantile(tab)
    qt_mil = fit_quantile(mil)

    t = qt_tab.transform(tab.reshape(-1, 1)).reshape(-1)
    m = qt_mil.transform(mil.reshape(-1, 1)).reshape(-1)

    X = make_meta_features(t, m)

    meta = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=seed,
        max_iter=5000
    )
    meta.fit(X, y)

    return meta, qt_tab, qt_mil


def apply_meta(meta, qt_tab, qt_mil, tab_pred, mil_pred):
    tab_pred = np.asarray(tab_pred, dtype=float).reshape(-1, 1)
    mil_pred = np.asarray(mil_pred, dtype=float).reshape(-1, 1)

    t = qt_tab.transform(tab_pred).reshape(-1)
    m = qt_mil.transform(mil_pred).reshape(-1)

    X = make_meta_features(t, m)
    return meta.predict_proba(X)[:, 1]


def build_merged_oof(tab_csv, mil_csv, out_dir, prefer_rank=True):
    os.makedirs(out_dir, exist_ok=True)

    tab = pd.read_csv(tab_csv)
    mil = pd.read_csv(mil_csv)

    # pick columns robustly
    tab_col = pick_col(
        tab,
        ["oof_rank", "oof_raw", "oof_p", "oof_prob"],
        "tabular OOF prediction column"
    )
    if prefer_rank and "oof_rank" in tab.columns:
        tab_col = "oof_rank"   

    mil_col = pick_col(mil, ["mil_prob", "oof_prob", "prob"], "MIL prediction column")

    # merge on subjectID
    df = tab.merge(mil[["subjectID", mil_col]], on="subjectID", how="inner").dropna().reset_index(drop=True)

    if "y" not in df.columns:
        if "y_x" in df.columns:
            df["y"] = df["y_x"]
        else:
            raise KeyError("No label column 'y' found after merge.")

    df = df.rename(columns={tab_col: "tab_pred", mil_col: "mil_pred"})
    df[["y"]] = df[["y"]].astype(int)

    y = df["y"].values
    m_tab = compute_metrics(y, df["tab_pred"].values)
    m_mil = compute_metrics(y, df["mil_pred"].values)


    oof_meta, qt_tab, qt_mil = train_stack_oof(df, "tab_pred", "mil_pred", n_splits=5)
    m_stack = compute_metrics(y, oof_meta)

    meta, qt_tab_final, qt_mil_final = train_final_meta(df, "tab_pred", "mil_pred")

    summary = {
        "rows_merged": int(len(df)),
        "base_tab": m_tab,
        "base_mil": m_mil,
        "stack_oof_eval": m_stack,
        "tab_used": "tab_pred",
        "mil_used": "mil_pred",
        "notes": "QuantileTransformer(CDF) + LogisticRegression stacking on [t, m, |t-m|, t*m]."
    }

    df_out = df[["subjectID", "y", "tab_pred", "mil_pred"]].copy()
    df_out["stack_oof"] = oof_meta
    df_out.to_csv(os.path.join(out_dir, "merged_oof_with_stack.csv"), index=False)

    with open(os.path.join(out_dir, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save meta artifacts 
    import pickle
    with open(os.path.join(out_dir, "meta_model.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(out_dir, "qt_tab.pkl"), "wb") as f:
        pickle.dump(qt_tab_final, f)
    with open(os.path.join(out_dir, "qt_mil.pkl"), "wb") as f:
        pickle.dump(qt_mil_final, f)

    print(json.dumps(summary, indent=2))
    print(f"Saved in: {out_dir}")


if __name__ == "__main__":
    # ---- PVH ----
    # Build with your real paths:
    build_merged_oof("/workspace/final_neckvibe/pvh_out/pvh_oof.csv",
                     "/workspace/final_neckvibe/mil_pvh_enhanced_out/mil_pvh_oof.csv",
                     "/workspace/final_neckvibe/pvh_ensemble_final",
                     prefer_rank=False)  

    # ---- NPVH ----
    build_merged_oof("/workspace/final_neckvibe/npvh_out_final/npvh_oof_final.csv",
                     "/workspace/final_neckvibe/mil_npvh_out_improved/mil_npvh_oof.csv",
                     "/workspace/final_neckvibe/npvh_ensemble_final",
                     prefer_rank=True)   
    pass
