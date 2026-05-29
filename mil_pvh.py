import os
import random
import math
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================
# CONFIG
# =========================
TRAIN_CSV   = "/workspace/NeckVibeChallenge/Labels/Train.csv"
FEATURE_DIR = "/workspace/NeckVibeChallenge/Features"
OUT_DIR     = "./mil_pvh_enhanced_out"

TASK_POS_LABEL = "PVH"
EXCLUDE_SINGING = True
FOLDS = 5
SEED = 42

# ---- MIL settings ---- #
WIN_FRAMES = 240                  # 12s (was 600)
MAX_WINDOWS_PER_SUBJECT = 24      # (was 32)
MIN_VOICED_FRAMES = 80            # (was 120)

USE_DAY_STATS = True
USE_CHANNEL_NORM = True

EPOCHS = 14                       # (was 30)
BATCH_SIZE = 16                   # (was 8)
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
EMBED_DIM = 256
ATTN_DIM = 128
EARLY_STOPPING_PATIENCE = 4

# DataLoader
NUM_WORKERS = 2                   # (was 4)
PIN_MEMORY = True
PERSISTENT_WORKERS = False       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

CHANNEL_KEYS = [
    "H1H2all", "LHratioall", "cppall", "dBcms2", "spectralTiltall", "level",
    "IBIF_h1h2", "IBIF_hrf", "IBIF_mfdr", "IBIF_acflow", "IBIF_sq",
    "IBIF_oq", "IBIF_naq", "IBIF_cq",
]

STAT_FEATURES = ['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'median', 'q25', 'q75']

# Cache: mat_path -> 
MAT_CACHE = {}
MAX_CACHE_ITEMS = 4000
# =========================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_statistics(x):
    """Compute robust statistics for a 1D array."""
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    feat = np.zeros(9, dtype=np.float32)
    if x.size == 0:
        return feat

    feat[0] = float(np.mean(x))
    feat[1] = float(np.std(x)) if x.size > 1 else 0.0
    if x.size > 2:
        feat[2] = float(sp_skew(x))
        feat[3] = float(sp_kurtosis(x))
    feat[4] = float(np.min(x))
    feat[5] = float(np.max(x))
    feat[6] = float(np.median(x))
    if x.size > 3:
        feat[7] = float(np.percentile(x, 25))
        feat[8] = float(np.percentile(x, 75))
    return feat


def robust_normalize(x):
    """Robust normalization using median and IQR."""
    x = np.asarray(x, dtype=np.float32)
    valid = x[np.isfinite(x)]
    if valid.size < 10:
        return x.astype(np.float32)

    med = np.median(valid)
    iqr = np.percentile(valid, 75) - np.percentile(valid, 25)
    if iqr < 1e-6:
        s = np.std(valid)
        iqr = s if s > 1e-6 else 1.0

    x_norm = (x - med) / (iqr + 1e-6)
    x_norm = np.clip(x_norm, -5, 5)
    return x_norm.astype(np.float32)


def build_mask(mat, exclude_singing=True):
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
    energy = get("level")

    lengths = []
    for v in [rec, voiced, sing, energy]:
        if v is not None:
            lengths.append(v.size)
    if not lengths:
        return None, None
    n = min(lengths)

    mask = np.ones(n, dtype=bool)
    if rec is not None:
        mask &= (rec[:n] > 0)
    if voiced is not None:
        mask &= (voiced[:n] > 0)
    if exclude_singing and sing is not None:
        mask &= (sing[:n] <= 0)

    if energy is not None:
        e = energy[:n].astype(np.float32)
        e_masked = e[mask]
        if e_masked.size > 20:
            mu = float(np.median(e_masked))
            sd = float(np.std(e_masked))
            if sd > 1e-6:
                z = np.abs((e - mu) / (sd + 1e-6))
                mask &= (z < 3.0)

    return mask, (energy[:n] if energy is not None else None)


def load_multichannel_enhanced(mat_path, exclude_singing=True):
    mat = loadmat(mat_path)
    mask, energy = build_mask(mat, exclude_singing=exclude_singing)
    if mask is None or int(mask.sum()) < MIN_VOICED_FRAMES:
        return None, None, None

    idx = np.where(mask)[0]
    n_mask = mask.size
    T = idx.size

    C = len(CHANNEL_KEYS)
    X = np.zeros((C, T), dtype=np.float32)
    miss = np.zeros((C,), dtype=np.float32)
    S = np.zeros((C, len(STAT_FEATURES)), dtype=np.float32) if USE_DAY_STATS else None

    for i, k in enumerate(CHANNEL_KEYS):
        if k not in mat:
            miss[i] = 1.0
            continue

        x = np.asarray(mat[k]).squeeze().reshape(-1)
        if x.size == 0:
            miss[i] = 1.0
            continue

        n = min(x.size, n_mask)
        x = x[:n].astype(np.float32)

        idx2 = idx[idx < n]
        if idx2.size < MIN_VOICED_FRAMES:
            miss[i] = 1.0
            continue

        x_voiced = x[idx2]
        x_voiced[~np.isfinite(x_voiced)] = 0.0

        if USE_CHANNEL_NORM and x_voiced.size > 10:
            x_voiced = robust_normalize(x_voiced)

        # pad/truncate to T
        if x_voiced.size < T:
            x_voiced = np.pad(x_voiced, (0, T - x_voiced.size))
        elif x_voiced.size > T:
            x_voiced = x_voiced[:T]

        X[i] = x_voiced

        if USE_DAY_STATS and S is not None:
            S[i] = compute_statistics(x_voiced)

    # Add energy channel (if available)
    if energy is not None:
        e = energy[idx].astype(np.float32)
        e[~np.isfinite(e)] = 0.0
        if USE_CHANNEL_NORM and e.size > 10:
            e = robust_normalize(e)

        if e.size < T:
            e = np.pad(e, (0, T - e.size))
        elif e.size > T:
            e = e[:T]

        X = np.vstack([X, e[np.newaxis, :]])
        miss = np.append(miss, 0.0)

        if USE_DAY_STATS and S is not None:
            S = np.vstack([S, compute_statistics(e)[np.newaxis, :]])

    return X, miss, S


def load_multichannel_enhanced_cached(mat_path, exclude_singing=True):
    key = (mat_path, exclude_singing)
    if key in MAT_CACHE:
        return MAT_CACHE[key]
    X, miss, S = load_multichannel_enhanced(mat_path, exclude_singing=exclude_singing)
    if len(MAT_CACHE) < MAX_CACHE_ITEMS:
        MAT_CACHE[key] = (X, miss, S)
    return X, miss, S


def sample_windows(X, miss, stats, win_frames, max_windows):
    C, T = X.shape
    if T < win_frames:
        X = np.pad(X, ((0, 0), (0, win_frames - T)), mode="edge")
        T = win_frames

    max_start = T - win_frames
    if max_start <= 0:
        max_start = 0

    # random diverse starts
    if max_start == 0:
        starts = [0] * max_windows
    else:
        # pick without replacement up to max_windows
        n_pick = min(max_windows, max_start + 1)
        starts = np.random.choice(max_start + 1, n_pick, replace=False).tolist()
        while len(starts) < max_windows:
            starts.append(int(np.random.randint(0, max_start + 1)))

    miss_rate = float(np.mean(miss))
    miss_ch = np.full((1, win_frames), miss_rate, dtype=np.float32)

    windows = []
    for s in starts[:max_windows]:
        w = X[:, s:s + win_frames]
        w = np.concatenate([w, miss_ch], axis=0)  
        windows.append(w)

    windows = np.stack(windows, axis=0)  # (K, C+1, T)

    # Add day stats as extra channels (broadcast over time)
    if stats is not None:
        # stats: (Cstat, 9) -> create (Cstat*9, T) constant channels
        flat = stats.reshape(-1).astype(np.float32)  # (Cstat*9,)
        stat_ch = np.repeat(flat[:, None], win_frames, axis=1)  # (Cstat*9, T)
        stat_ch = np.repeat(stat_ch[None, :, :], max_windows, axis=0)  # (K, Cstat*9, T)
        windows = np.concatenate([windows, stat_ch], axis=1)

    return windows


class EnhancedSubjectBagDataset(Dataset):
    def __init__(self, subject_to_paths, subject_to_label, subject_ids,
                 exclude_singing=True, win_frames=240, max_windows=24):
        self.subject_to_paths = subject_to_paths
        self.subject_to_label = subject_to_label
        self.subject_ids = subject_ids
        self.exclude_singing = exclude_singing
        self.win_frames = win_frames
        self.max_windows = max_windows

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        paths = self.subject_to_paths[sid]
        y = float(self.subject_to_label[sid])

        per_day = max(1, int(math.ceil(self.max_windows / max(1, len(paths)))))

        windows_all = []
        for p in paths:
            X, miss, S = load_multichannel_enhanced_cached(p, exclude_singing=self.exclude_singing)
            if X is None:
                continue
            w = sample_windows(X, miss, S, self.win_frames, per_day)
            windows_all.append(w)

        # Determine final channel count dynamically
        if len(windows_all) == 0:
            # fallback
            base_c = len(CHANNEL_KEYS) + 1 + 1  
            stat_c = ( (len(CHANNEL_KEYS) + 1) * len(STAT_FEATURES) ) if USE_DAY_STATS else 0
            Ctot = base_c + stat_c
            bag = np.zeros((self.max_windows, Ctot, self.win_frames), dtype=np.float32)
        else:
            bag = np.concatenate(windows_all, axis=0)  # (K', Ctot, T)
            if bag.shape[0] >= self.max_windows:
                bag = bag[:self.max_windows]
            else:
                pad_k = self.max_windows - bag.shape[0]
                pad = np.zeros((pad_k, bag.shape[1], bag.shape[2]), dtype=np.float32)
                bag = np.concatenate([bag, pad], axis=0)

        return torch.tensor(bag, dtype=torch.float32), torch.tensor([y], dtype=torch.float32), sid


# -------------------------
# Model
# -------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResBlock1D_SE(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dropout=0.3, use_se=True):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.skip is not None:
            identity = self.skip(identity)
        return self.act(out + identity)


class WindowEncoder(nn.Module):
    def __init__(self, in_ch, emb=256, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block1 = ResBlock1D_SE(64, 64, k=7, stride=1, dropout=dropout)
        self.block2 = ResBlock1D_SE(64, 128, k=7, stride=2, dropout=dropout)
        self.block3 = ResBlock1D_SE(128, 256, k=7, stride=2, dropout=dropout)
        self.block4 = ResBlock1D_SE(256, 256, k=7, stride=1, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, emb),
            nn.LayerNorm(emb),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class GatedAttentionMIL(nn.Module):
    def __init__(self, emb=256, attn=128, dropout=0.3):
        super().__init__()
        self.V = nn.Linear(emb, attn)
        self.U = nn.Linear(emb, attn)
        self.w = nn.Linear(attn, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, Z):
        A_V = self.tanh(self.V(self.drop(Z)))
        A_U = self.sigmoid(self.U(self.drop(Z)))
        A = self.w(A_V * A_U)
        w = torch.softmax(A, dim=1)
        s = torch.sum(w * Z, dim=1)
        return s, w.squeeze(-1)


class MILClassifier(nn.Module):
    def __init__(self, in_ch, emb=256, dropout=0.3):
        super().__init__()
        self.enc = WindowEncoder(in_ch=in_ch, emb=emb, dropout=dropout)
        self.mil = GatedAttentionMIL(emb=emb, attn=ATTN_DIM, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(emb, emb // 2),
            nn.LayerNorm(emb // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb // 2, emb // 4),
            nn.LayerNorm(emb // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb // 4, 1),
        )

    def forward(self, bag):
        B, K, C, T = bag.shape
        x = bag.view(B * K, C, T)
        z = self.enc(x).view(B, K, -1)
        s, w = self.mil(z)
        logit = self.head(s)
        return logit, w


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


def train_fold(fold, train_ids, val_ids, subject_to_paths, subject_to_label):
    tr_ds = EnhancedSubjectBagDataset(subject_to_paths, subject_to_label, train_ids,
                                     exclude_singing=EXCLUDE_SINGING,
                                     win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)
    va_ds = EnhancedSubjectBagDataset(subject_to_paths, subject_to_label, val_ids,
                                     exclude_singing=EXCLUDE_SINGING,
                                     win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                           persistent_workers=PERSISTENT_WORKERS)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    y_tr = np.array([subject_to_label[s] for s in train_ids], dtype=np.float32)
    pos = max(1.0, float(np.sum(y_tr)))
    neg = max(1.0, float((y_tr == 0).sum()))
    pos_weight = neg / pos

    # infer in_ch from one sample
    sample_bag, _, _ = tr_ds[0]
    in_ch = sample_bag.shape[1]

    model = MILClassifier(in_ch=in_ch, emb=EMBED_DIM, dropout=DROPOUT).to(DEVICE)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=1, eta_min=LR/50)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_auc = -1.0
    best_state = None
    patience = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for bag, y, _ in tr_loader:
            bag = bag.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logit, _ = model(bag)
                loss = bce(logit, y) + 0.05 * focal(logit, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))

        sched.step()

        # val
        model.eval()
        p_all, y_all = [], []
        with torch.no_grad():
            for bag, y, _ in va_loader:
                bag = bag.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit, _ = model(bag)
                    p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
                p_all.append(p)
                y_all.append(y.numpy().reshape(-1))

        p_all = np.concatenate(p_all) if p_all else np.array([], dtype=np.float32)
        y_all = np.concatenate(y_all) if y_all else np.array([], dtype=np.float32)
        val_auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else 0.5
        val_ap = average_precision_score(y_all, p_all) if len(np.unique(y_all)) > 1 else 0.0

        print(f"[PVH][Fold {fold}] Ep {ep:02d} loss={np.mean(losses):.4f} val_auc={val_auc:.4f} ap={val_ap:.4f}")

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"[PVH][Fold {fold}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc


def main():
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("DEVICE =", DEVICE, "| AMP =", USE_AMP)

    meta = pd.read_csv(TRAIN_CSV)

    # find filename column robustly
    mat_col = None
    for c in ["Feature filename", "feature filename", "feat_path", "mat_path", "file", "filename"]:
        if c in meta.columns:
            mat_col = c
            break
    if mat_col is None:
        for c in meta.columns:
            if "filename" in c.lower() or "mat" in c.lower() or "file" in c.lower():
                mat_col = c
                break
    if mat_col is None:
        raise ValueError(f"Could not find filename column. Available: {list(meta.columns)}")

    # subject -> paths
    subject_to_paths = {}
    missing = 0
    for _, r in meta.iterrows():
        sid = str(r["subjectID"])
        mp = str(r[mat_col]).strip()
        if mp.lower() in ["nan", "none", ""]:
            continue
        mat_path = mp if os.path.isabs(mp) else os.path.join(FEATURE_DIR, mp)
        if not os.path.exists(mat_path):
            missing += 1
            continue
        subject_to_paths.setdefault(sid, []).append(mat_path)

    # labels
    y_day = (meta["groupLabel"].astype(str) == TASK_POS_LABEL).astype(int).values
    tmp = pd.DataFrame({"subjectID": meta["subjectID"].astype(str), "y": y_day})
    subj_y = tmp.groupby("subjectID")["y"].max().to_dict()
    subject_to_label = {sid: int(subj_y.get(sid, 0)) for sid in subject_to_paths.keys()}

    subject_ids = sorted(list(subject_to_paths.keys()))
    y_subj = np.array([subject_to_label[s] for s in subject_ids], dtype=int)
    groups = np.array(subject_ids)

    print(f"[PVH] Subjects={len(subject_ids)} Pos={int(y_subj.sum())} Missing_skipped={missing}")

    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof = {}
    fold_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros_like(y_subj), y_subj, groups), 1):
        train_ids = [subject_ids[i] for i in tr_idx]
        val_ids = [subject_ids[i] for i in va_idx]

        model, best_auc = train_fold(fold, train_ids, val_ids, subject_to_paths, subject_to_label)
        fold_aucs.append(best_auc)

        # save fold model
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"mil_pvh_fold{fold}.pt"))

        # val predictions
        val_ds = EnhancedSubjectBagDataset(subject_to_paths, subject_to_label, val_ids,
                                           exclude_singing=EXCLUDE_SINGING,
                                           win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model.eval()
        probs = {}
        with torch.no_grad():
            for bag, y, sids in val_loader:
                bag = bag.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit, _ = model(bag)
                    p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
                for sid, pv in zip(sids, p):
                    probs[sid] = float(pv)

        # store OOF
        for sid in val_ids:
            oof[sid] = probs.get(sid, 0.5)

        print(f"[PVH] Fold {fold}: best_val_auc={best_auc:.4f}")

    # overall
    oof_sids = list(oof.keys())
    y_true = np.array([subject_to_label[s] for s in oof_sids], dtype=np.float32)
    y_pred = np.array([oof[s] for s in oof_sids], dtype=np.float32)

    oof_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    oof_ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    print(f"\n[PVH] Fold AUCs: {np.round(fold_aucs,4).tolist()} Mean={np.mean(fold_aucs):.4f}")
    print(f"[PVH] OOF AUC: {oof_auc:.4f} | OOF AP: {oof_ap:.4f}")

    pd.DataFrame({
        "subjectID": oof_sids,
        "y": [subject_to_label[s] for s in oof_sids],
        "mil_prob": [oof[s] for s in oof_sids],
    }).to_csv(os.path.join(OUT_DIR, "mil_pvh_oof.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "fold_aucs": [float(x) for x in fold_aucs],
            "oof_auc": float(oof_auc),
            "oof_ap": float(oof_ap),
            "config": {
                "WIN_FRAMES": WIN_FRAMES,
                "MAX_WINDOWS_PER_SUBJECT": MAX_WINDOWS_PER_SUBJECT,
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "DROPOUT": DROPOUT,
                "USE_DAY_STATS": USE_DAY_STATS,
                "USE_CHANNEL_NORM": USE_CHANNEL_NORM,
                "DEVICE": DEVICE,
                "USE_AMP": USE_AMP
            }
        }, f, indent=2)

    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
