"""
Baseline models for LSST classification.
B2: MultiROCKET (aeon, no pre-training)
B3: 1D-CNN from scratch (PyTorch)
"""

import numpy as np
import torch
import torch.nn as nn
from aeon.classification.convolution_based import MultiRocketClassifier

SEED = 42


# ──────────────────────────────────────────────
# B2: MultiROCKET
# ──────────────────────────────────────────────

def train_multirocket(X_train, y_train, n_kernels=10_000):
    """Train MultiROCKET. X must be (n, C, T) — channels-first."""
    clf = MultiRocketClassifier(n_kernels=n_kernels, random_state=SEED)
    clf.fit(X_train, y_train)
    return clf


# ──────────────────────────────────────────────
# B3: 1D-CNN from scratch
# ──────────────────────────────────────────────

class CNN1D(nn.Module):
    """Simple 3-layer 1D-CNN for multivariate time series classification."""

    def __init__(self, n_channels=6, seq_len=36, n_classes=14):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)  # (batch, 256)
        return self.classifier(x)


def train_cnn(X_train, y_train, X_val, y_val, class_weights,
              n_channels=6, seq_len=36, n_classes=14,
              epochs=100, lr=1e-3, batch_size=128, patience=15,
              device="cuda"):
    """Train the 1D-CNN with class-weighted loss, cosine LR, early stopping.
    X arrays should be numpy (n, C, T). Returns (model, history dict).
    """
    model = CNN1D(n_channels, seq_len, n_classes).to(device)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # tensors
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / n_batches)

        # --- val ---
        model.eval()
        with torch.no_grad():
            logits_v = model(Xv)
            val_loss = criterion(logits_v, yv).item()
            val_acc = (logits_v.argmax(1) == yv).float().mean().item()
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} — train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, history
