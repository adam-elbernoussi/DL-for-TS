"""
Centralized data pipeline for the LSST classification project.
Loads LSST from UCR/UEA, provides channels-first format, MOMENT padding,
label encoding, class weights, and stratified splits.
"""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tslearn.datasets import UCR_UEA_datasets

SEED = 42
MOMENT_SEQ_LEN = 512   # MOMENT expects 512 timesteps
MOMENT_PATCH_LEN = 8   # MOMENT patch length
# 36 timesteps → pad to 40 (nearest multiple of 8) to get 5 full patches
# instead of 4 full patches + 4 discarded timesteps (11% data loss)
LSST_REAL_LEN = 36
LSST_PATCH_ALIGNED_LEN = 40  # ceil(36/8)*8


def load_lsst_raw():
    """Load raw LSST dataset. Returns (X_train, y_train, X_test, y_test)
    with X shape (n_samples, 36, 6) — tslearn default (samples, time, channels).
    """
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")
    return X_train, y_train, X_test, y_test


def to_channels_first(X):
    """(n, T, C) -> (n, C, T)"""
    return np.transpose(X, (0, 2, 1))


def encode_labels(y_train, y_test):
    """Encode string labels to integers. Returns encoded arrays + fitted encoder."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def compute_class_weights(y_encoded):
    """Compute balanced class weights as a numpy array."""
    classes = np.unique(y_encoded)
    weights = compute_class_weight("balanced", classes=classes, y=y_encoded)
    return weights


def get_class_weight_tensor(y_encoded, device="cpu"):
    """Return class weights as a torch tensor for CrossEntropyLoss."""
    weights = compute_class_weights(y_encoded)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def pad_for_moment(X_cf, seq_len=MOMENT_SEQ_LEN):
    """Pad channels-first data (n, C, T) to (n, C, seq_len).
    Pads to LSST_PATCH_ALIGNED_LEN first (40 timesteps = 5 full patches of 8),
    then zero-pads to seq_len=512.
    input_mask marks the patch-aligned region (40 timesteps) as real, rest as padding.
    """
    n, c, t = X_cf.shape
    real_len = min(LSST_PATCH_ALIGNED_LEN, seq_len)  # 40
    X_padded = np.zeros((n, c, seq_len), dtype=np.float32)
    X_padded[:, :, :t] = X_cf  # place real data at start (t=36)
    # mask covers 40 timesteps (5 full patches), not just 36
    input_mask = np.zeros((n, seq_len), dtype=np.float32)
    input_mask[:, :real_len] = 1.0
    return X_padded, input_mask


def global_standardize(X_train_cf, X_test_cf):
    """Globally standardize per channel using train statistics.
    This preserves amplitude differences across samples (unlike per-instance RevIN),
    so that class-discriminative brightness information survives MOMENT's RevIN.
    X: (n, C, T) channels-first.
    """
    # Compute mean/std per channel over all train samples and timesteps
    mean = X_train_cf.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std  = X_train_cf.std(axis=(0, 2), keepdims=True).clip(min=1e-8)
    X_train_scaled = (X_train_cf - mean) / std
    X_test_scaled  = (X_test_cf  - mean) / std
    return X_train_scaled, X_test_scaled, mean, std


def stratified_train_val_split(X, y, val_size=0.2, seed=SEED):
    """Stratified split into train/val. Works with any X shape."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )
    return X_tr, X_val, y_tr, y_val


def load_lsst_for_sklearn():
    """Load LSST in channels-first format with encoded labels.
    Returns dict with keys: X_train, X_test, y_train, y_test,
    label_encoder, class_weights, n_classes, class_names.
    """
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_lsst_raw()
    X_train = to_channels_first(X_train_raw).astype(np.float32)
    X_test = to_channels_first(X_test_raw).astype(np.float32)
    y_train, y_test, le = encode_labels(y_train_raw, y_test_raw)
    cw = compute_class_weights(y_train)
    return dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        label_encoder=le,
        class_weights=cw,
        n_classes=len(le.classes_),
        class_names=le.classes_,
    )


def load_lsst_for_moment(device="cpu"):
    """Load LSST padded to 512 for MOMENT, as torch tensors.
    Applies global per-channel standardization before padding so that
    amplitude differences (class-discriminative in LSST) survive MOMENT's RevIN.
    The input_mask covers 40 timesteps (5 full patches of 8) so no data is discarded.
    """
    data = load_lsst_for_sklearn()
    # Global standardization: preserves relative amplitude across samples
    X_train_sc, X_test_sc, _, _ = global_standardize(data["X_train"], data["X_test"])
    # Pad to 512 (mask covers first 40 timesteps)
    X_train_pad, mask_train = pad_for_moment(X_train_sc)
    X_test_pad, mask_test = pad_for_moment(X_test_sc)
    cw_tensor = torch.tensor(data["class_weights"], dtype=torch.float32, device=device)
    return dict(
        X_train=torch.tensor(X_train_pad, device=device),
        X_test=torch.tensor(X_test_pad, device=device),
        mask_train=torch.tensor(mask_train, device=device),
        mask_test=torch.tensor(mask_test, device=device),
        y_train=torch.tensor(data["y_train"], dtype=torch.long, device=device),
        y_test=torch.tensor(data["y_test"], dtype=torch.long, device=device),
        label_encoder=data["label_encoder"],
        class_weights_tensor=cw_tensor,
        n_classes=data["n_classes"],
        class_names=data["class_names"],
    )
