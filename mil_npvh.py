import os
import random
import math
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

# =========================
# CONFIG 
# =========================
TRAIN_CSV   = "/workspace/NeckVibeChallenge/Labels/Train.csv"
FEATURE_DIR = "/workspace/NeckVibeChallenge/Features"
OUT_DIR     = "./mil_npvh_out_improved"

TASK_POS_LABEL = "NPVH"
EXCLUDE_SINGING = True
FOLDS = 5
SEED = 42

# ---- MIL settings ---- #
WIN_FRAMES = 240                  # 12 sec (stable + fast)
MAX_WINDOWS_PER_SUBJECT = 24      # K
MIN_VOICED_FRAMES = 80

EPOCHS = 16
BATCH_SIZE = 16
LR = 7e-4                        
WEIGHT_DECAY = 2e-4
DROPOUT = 0.30
EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# Channels
CHANNEL_KEYS = [
    "cppall", "H1H2all", "spectralTiltall", "level", "LHratioall", "fo", "dBcms2",
    "IBIF_hrf", "IBIF_h1h2", "IBIF_mfdr"
]

# Cache mat loads
MAT_CACHE = {}
MAX_CACHE_ITEMS = 4000
# =========================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_1d(arr):
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def rank_normalize(p):
    p = np.asarray(p, dtype=np.float32)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(p), dtype=np.float32)
    return ranks / max(1.0, (len(p) - 1.0))


def build_mask(mat, exclude_singing=True):
    def get(k):
        if k not in mat:
            return None
        x = _as_1d(mat[k])
        return x

    rec = get("recordingOn")
    voiced = get("voiced")
    sing = get("voiced_singing") if exclude_singing else None

    lengths = []
    for v in [rec, voiced, sing]:
        if v is not None and v.size > 0:
            lengths.append(v.size)
    if not lengths:
        return None

    n = min(lengths)
    mask = np.ones(n, dtype=bool)
    if rec is not None:
        mask &= (rec[:n] > 0)
    if voiced is not None:
        mask &= (voiced[:n] > 0)
    if exclude_singing and sing is not None:
        mask &= (sing[:n] <= 0)
    return mask


def robust_norm(x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    v = x[np.isfinite(x)]
    if v.size < 20:
        return x
    med = np.median(v)
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    if iqr < 1e-6:
        sd = np.std(v)
        iqr = sd if sd > 1e-6 else 1.0
    z = (x - med) / (iqr + 1e-6)
    z = np.clip(z, -6, 6)
    return z.astype(np.float32)


def load_multichannel(mat_path, exclude_singing=True):
    mat = loadmat(mat_path)
    mask = build_mask(mat, exclude_singing=exclude_singing)
    if mask is None or int(mask.sum()) < MIN_VOICED_FRAMES:
        return None, None

    n_mask = mask.size
    idx = np.where(mask)[0]
    T = idx.size
    C = len(CHANNEL_KEYS)

    X = np.zeros((C, T), dtype=np.float32)
    miss = np.zeros((C,), dtype=np.float32)

    for i, k in enumerate(CHANNEL_KEYS):
        if k not in mat:
            miss[i] = 1.0
            continue

        x = _as_1d(mat[k])
        if x.size <= 0:
            miss[i] = 1.0
            continue

        n = min(x.size, n_mask)
        x = x[:n]
        idx2 = idx[idx < n]

        if idx2.size < MIN_VOICED_FRAMES:
            miss[i] = 1.0
            continue

        x = x[idx2].astype(np.float32)
        x = robust_norm(x)

        if x.size < T:
            x = np.pad(x, (0, T - x.size))
        elif x.size > T:
            x = x[:T]
        X[i] = x

    return X, miss


def load_multichannel_cached(mat_path, exclude_singing=True):
    key = (mat_path, exclude_singing)
    if key in MAT_CACHE:
        return MAT_CACHE[key]
    X, miss = load_multichannel(mat_path, exclude_singing=exclude_singing)
    if len(MAT_CACHE) < MAX_CACHE_ITEMS:
        MAT_CACHE[key] = (X, miss)
    return X, miss


def sample_windows_diverse(X, miss, win_frames, max_windows):
    C, T = X.shape
    if T < win_frames:
        X = np.pad(X, ((0, 0), (0, win_frames - T)), mode="edge")
        T = win_frames

    max_start = T - win_frames
    miss_rate = float(np.mean(miss))
    miss_ch = np.full((1, win_frames), miss_rate, dtype=np.float32)

    windows = []

    # 60% random unique starts
    n_rand = int(max_windows * 0.6)
    if max_start > 0 and n_rand > 0:
        picks = min(n_rand, max_start + 1)
        starts = np.random.choice(max_start + 1, picks, replace=False).tolist()
        while len(starts) < n_rand:
            starts.append(int(np.random.randint(0, max_start + 1)))
        for s in starts[:n_rand]:
            w = X[:, s:s+win_frames]
            windows.append(np.concatenate([w, miss_ch], axis=0))

    # 20% high-|CPP| region (NPVH often quality-related)
    n_cpp = int(max_windows * 0.2)
    if n_cpp > 0 and "cppall" in CHANNEL_KEYS and max_start > 0:
        ci = CHANNEL_KEYS.index("cppall")
        sig = X[ci]
        # pick starts around lowest cpp (degraded quality) and highest (contrast)
        order = np.argsort(sig)
        cand = np.concatenate([order[:max(1, len(order)//8)], order[-max(1, len(order)//8):]])
        cand = cand[cand <= max_start]
        if cand.size > 0:
            pick = min(n_cpp, cand.size)
            ssel = np.random.choice(cand, pick, replace=False)
            for s in ssel:
                w = X[:, s:s+win_frames]
                windows.append(np.concatenate([w, miss_ch], axis=0))

    # remainder: uniform coverage
    while len(windows) < max_windows:
        if max_start <= 0:
            s = 0
        else:
            step = max(1, max_start // max(1, (max_windows - len(windows))))
            s = min(max_start, (len(windows) * step) % (max_start + 1))
        w = X[:, s:s+win_frames]
        windows.append(np.concatenate([w, miss_ch], axis=0))

    windows = windows[:max_windows]
    return np.stack(windows, axis=0)  # (K, C+1, T)


class SubjectBagDataset(Dataset):
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

        all_w = []
        for p in paths:
            X, miss = load_multichannel_cached(p, exclude_singing=self.exclude_singing)
            if X is None:
                continue
            w = sample_windows_diverse(X, miss, self.win_frames, per_day)
            all_w.append(w)

        C1 = len(CHANNEL_KEYS) + 1
        if len(all_w) == 0:
            bag = np.zeros((self.max_windows, C1, self.win_frames), dtype=np.float32)
        else:
            bag = np.concatenate(all_w, axis=0)
            if bag.shape[0] >= self.max_windows:
                bag = bag[:self.max_windows]
            else:
                pad = np.zeros((self.max_windows - bag.shape[0], C1, self.win_frames), dtype=np.float32)
                bag = np.concatenate([bag, pad], axis=0)

        return torch.tensor(bag, dtype=torch.float32), torch.tensor([y], dtype=torch.float32), sid


# -------------------------
# Model: ResNet1D + Attn MIL
# -------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dropout=0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        return self.act(out + identity)


class WindowEncoder(nn.Module):
    def __init__(self, in_ch, emb=128, dropout=0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.b1 = ResBlock1D(64, 64, k=7, stride=1, dropout=dropout)
        self.b2 = ResBlock1D(64, 128, k=7, stride=2, dropout=dropout)
        self.b3 = ResBlock1D(128, 128, k=7, stride=1, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, emb)

    def forward(self, x):
        h = self.stem(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class GatedAttentionMIL(nn.Module):
    def __init__(self, emb=128, attn=128, dropout=0.30):
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
    def __init__(self, in_ch, emb=128, dropout=0.30):
        super().__init__()
        self.enc = WindowEncoder(in_ch=in_ch, emb=emb, dropout=dropout)
        self.mil = GatedAttentionMIL(emb=emb, attn=emb, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Dropout(dropout),
            nn.Linear(emb, 1)
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
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs = {}
    ys = {}
    for bag, y, sid in loader:
        bag = bag.to(DEVICE, non_blocking=True)
        logit, _ = model(bag)
        p = torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)
        for i, s in enumerate(sid):
            probs[s] = float(p[i])
            ys[s] = float(y[i].item())
    return probs, ys


def make_sampler(subject_ids, subject_to_label):
    labels = np.array([subject_to_label[s] for s in subject_ids], dtype=np.float32)
    pos = max(1.0, labels.sum())
    neg = max(1.0, (labels == 0).sum())
    w_pos = neg / pos
    weights = np.where(labels > 0, w_pos, 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(subject_ids), replacement=True)
    return sampler


def train_fold(fold, train_ids, val_ids, subject_to_paths, subject_to_label):
    tr_ds = SubjectBagDataset(subject_to_paths, subject_to_label, train_ids,
                              exclude_singing=EXCLUDE_SINGING,
                              win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)
    va_ds = SubjectBagDataset(subject_to_paths, subject_to_label, val_ids,
                              exclude_singing=EXCLUDE_SINGING,
                              win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)

    sampler = make_sampler(train_ids, subject_to_label)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)

    ytr = np.array([subject_to_label[s] for s in train_ids], dtype=np.float32)
    pos = max(1.0, float(ytr.sum()))
    neg = max(1.0, float((ytr == 0).sum()))
    pos_weight = neg / pos

    in_ch = len(CHANNEL_KEYS) + 1
    model = MILClassifier(in_ch=in_ch, emb=128, dropout=DROPOUT).to(DEVICE)

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
                loss = bce(logit, y) + 0.20 * focal(logit, y)  # stronger focal helps NPVH

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))

        sched.step()

        probs, ys = predict_probs(model, va_loader)
        y_true = np.array([ys[s] for s in probs.keys()], dtype=np.float32)
        y_pred = np.array([probs[s] for s in probs.keys()], dtype=np.float32)

        val_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        print(f"[NPVH][Fold {fold}] Epoch {ep:02d} loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"[NPVH][Fold {fold}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc


def main():
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("DEVICE =", DEVICE, "| AMP =", USE_AMP)

    meta = pd.read_csv(TRAIN_CSV)

    # filename column
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
        raise ValueError(f"Cannot find filename column. Available: {list(meta.columns)}")

    y_day = (meta["groupLabel"].astype(str) == TASK_POS_LABEL).astype(int).values

    subject_to_paths = {}
    missing = 0
    for i, r in meta.iterrows():
        sid = str(r["subjectID"])
        mp = str(r[mat_col]).strip()
        mat_path = mp if os.path.isabs(mp) else os.path.join(FEATURE_DIR, mp)
        if not os.path.exists(mat_path):
            missing += 1
            continue
        subject_to_paths.setdefault(sid, []).append(mat_path)

    tmp = pd.DataFrame({"subjectID": meta["subjectID"].astype(str), "y": y_day})
    subj_y = tmp.groupby("subjectID")["y"].max().to_dict()
    subject_to_label = {sid: int(subj_y.get(sid, 0)) for sid in subject_to_paths.keys()}

    subject_ids = sorted(list(subject_to_paths.keys()))
    y_subj = np.array([subject_to_label[s] for s in subject_ids], dtype=int)
    groups = np.array(subject_ids)

    print(f"[NPVH] Subjects: {len(subject_ids)}   Pos: {int(y_subj.sum())}   Missing mats skipped: {missing}")

    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    oof = {}
    fold_aucs = []

    for fold, (tr, va) in enumerate(cv.split(np.zeros_like(y_subj), y_subj, groups), 1):
        train_ids = [subject_ids[i] for i in tr]
        val_ids   = [subject_ids[i] for i in va]

        model, best_auc = train_fold(fold, train_ids, val_ids, subject_to_paths, subject_to_label)
        fold_aucs.append(best_auc)

        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"mil_npvh_fold{fold}.pt"))

        # val predictions
        va_ds = SubjectBagDataset(subject_to_paths, subject_to_label, val_ids,
                                  exclude_singing=EXCLUDE_SINGING,
                                  win_frames=WIN_FRAMES, max_windows=MAX_WINDOWS_PER_SUBJECT)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        probs, _ = predict_probs(model, va_loader)
        val_sids = list(probs.keys())
        val_p = np.array([probs[s] for s in val_sids], dtype=np.float32)

        # fold-wise rank normalize for stable OOF
        val_p = rank_normalize(val_p)

        for i, sid in enumerate(val_sids):
            oof[sid] = float(val_p[i])

        print(f"[NPVH] Fold {fold}: best_val_auc={best_auc:.4f}")

    # overall OOF AUC
    oof_sids = list(oof.keys())
    y_true = np.array([subject_to_label[s] for s in oof_sids], dtype=np.float32)
    y_pred = np.array([oof[s] for s in oof_sids], dtype=np.float32)
    oof_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5

    print(f"\n[NPVH] Fold AUCs: {np.round(fold_aucs,4).tolist()}  Mean={np.mean(fold_aucs):.4f}")
    print(f"[NPVH] OOF AUC (rank-normalized): {oof_auc:.4f}")

    pd.DataFrame({
        "subjectID": oof_sids,
        "y": [subject_to_label[s] for s in oof_sids],
        "mil_prob": [oof[s] for s in oof_sids],
    }).to_csv(os.path.join(OUT_DIR, "mil_npvh_oof.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "fold_aucs": [float(x) for x in fold_aucs],
            "oof_auc": float(oof_auc),
            "config": {
                "WIN_FRAMES": WIN_FRAMES,
                "MAX_WINDOWS_PER_SUBJECT": MAX_WINDOWS_PER_SUBJECT,
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "DROPOUT": DROPOUT,
                "USE_AMP": USE_AMP,
                "EXCLUDE_SINGING": EXCLUDE_SINGING
            }
        }, f, indent=2)

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()


