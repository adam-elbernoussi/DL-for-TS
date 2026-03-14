"""
TiReX foundation model pipeline for LSST classification.

M4: Zero-shot embedding extraction + sklearn classifier
M5: Pre-computed features + trainable MLP head (encoder frozen)
M6: Full end-to-end fine-tuning

TiReX is a univariate forecasting FM (NeurIPS 2025 workshop, NX-AI/TiRex).
Each of the 6 LSST channels is treated as an independent univariate series.
We flatten (n, C, T) -> (n*C, T), embed, then reshape to (n, C*hidden_dim).

Architecture: patch_size=32, hidden_dim=512, 12 layers
36 timesteps -> 2 patches per channel -> 6*512 = 3072-dim embedding

Key: _embed_context() uses @torch.inference_mode(), so for fine-tuning (M6)
we call _forward_model() directly to preserve gradient flow.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tirex import load_model
from sklearn.metrics import f1_score

SEED = 42
TIREX_HIDDEN_DIM = 512


# ──────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────

def _embed_batch(backbone, x_flat, layer=-1):
    """Run TiReX encoder on a flat batch (bs, T) and return (bs, hidden_dim).
    Bypasses @torch.inference_mode() by calling _forward_model directly.
    Gradients flow through backbone parameters.
    """
    input_embeds, padded_token = backbone._prepare_context_for_embedding(x_flat, None)
    _, hidden_states = backbone._forward_model(input_embeds, return_all_hidden=True)
    # hidden_states: (bs, num_tokens, num_layers, hidden_dim)
    real = hidden_states[:, padded_token:, layer, :]  # (bs, n_real_tokens, hidden_dim)
    return real.mean(dim=1)                           # (bs, hidden_dim)


# ──────────────────────────────────────────────
# M4: Zero-shot embedding extraction
# ──────────────────────────────────────────────

def extract_tirex_embeddings(X_cf, model_name="NX-AI/TiRex",
                              batch_size=512, device="cuda", layer=-1):
    """Extract TiReX embeddings for multivariate classification.

    TiReX is univariate: each channel is embedded independently.
    Channels are flattened into the batch dimension, then reassembled.

    Args:
        X_cf: (n, C, T) channels-first numpy array or torch tensor
        model_name: HuggingFace model path
        batch_size: univariate series per forward pass
        device: compute device
        layer: which transformer layer to use (-1 = last, most semantic)

    Returns:
        numpy array of shape (n, C * hidden_dim) = (n, 3072) for TiRex + 6 channels
    """
    if isinstance(X_cf, torch.Tensor):
        X_cf = X_cf.cpu().numpy()

    n, C, T = X_cf.shape
    X_flat = X_cf.reshape(n * C, T).astype(np.float32)  # (n*C, T)

    model = load_model(model_name, device=device)
    model.eval()

    all_embeddings = []
    for i in range(0, n * C, batch_size):
        batch = torch.tensor(X_flat[i:i + batch_size])
        emb = model._embed_context(batch)   # uses inference_mode — fine for zero-shot
        emb_layer = emb[:, :, layer, :]     # (bs, num_tokens, hidden_dim)
        emb_pooled = emb_layer.mean(dim=1)  # (bs, hidden_dim)
        all_embeddings.append(emb_pooled.cpu().numpy())

    embeddings_flat = np.concatenate(all_embeddings, axis=0)  # (n*C, hidden_dim)
    hidden_dim = embeddings_flat.shape[-1]
    return embeddings_flat.reshape(n, C * hidden_dim)          # (n, C * hidden_dim)


# ──────────────────────────────────────────────
# M6: Full fine-tuning
# ──────────────────────────────────────────────

class TiRexClassifier(nn.Module):
    """TiReX backbone + MLP classification head for end-to-end fine-tuning.

    TiReX is univariate: channels are processed independently then concatenated.
    Forward pass calls _forward_model directly (bypassing @inference_mode)
    so gradients flow through all backbone parameters.
    """

    def __init__(self, backbone, n_classes, n_channels=6,
                 hidden_dim=TIREX_HIDDEN_DIM, layer=-1):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.layer = layer
        self.head = nn.Sequential(
            nn.Linear(n_channels * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def _encode(self, x_cf):
        """x_cf: (batch, n_channels, T) -> (batch, n_channels * hidden_dim)"""
        bs, C, T = x_cf.shape
        x_flat = x_cf.reshape(bs * C, T)
        emb = _embed_batch(self.backbone, x_flat, self.layer)  # (bs*C, hidden_dim)
        return emb.reshape(bs, C * emb.shape[-1])              # (bs, C * hidden_dim)

    def forward(self, x_cf):
        return self.head(self._encode(x_cf))


def load_tirex_for_classification(n_channels=6, n_classes=14,
                                   model_name="NX-AI/TiRex", device="cuda"):
    """Load TiReX wrapped with a classification head. All params trainable."""
    backbone = load_model(model_name, device=device)
    backbone.train()
    for param in backbone.parameters():
        param.requires_grad = True

    model = TiRexClassifier(backbone, n_classes=n_classes, n_channels=n_channels)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
    return model


def finetune_tirex(model, X_train, y_train, X_val, y_val,
                   class_weights, n_classes=14,
                   epochs=50, lr=1e-4, batch_size=32, patience=10,
                   device="cuda"):
    """Full fine-tune TiReX for classification (all params trainable).
    X_train/X_val: (n, C, T) numpy arrays (channels-first, raw data).
    Returns (model, history).
    """
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.long)

    train_ds = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_wf1": []}
    best_wf1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history["train_loss"].append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            all_logits = []
            for i in range(0, len(Xv), batch_size):
                xb = Xv[i:i + batch_size].to(device)
                all_logits.append(model(xb).cpu())
            logits_v = torch.cat(all_logits, dim=0)
            val_loss = criterion(logits_v.to(device), yv.to(device)).item()
            y_pred_v = logits_v.argmax(1).numpy()
            wf1 = f1_score(y_val, y_pred_v, average="weighted")

        history["val_loss"].append(val_loss)
        history["val_wf1"].append(wf1)

        if wf1 > best_wf1:
            best_wf1 = wf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch+1}/{epochs} — train_loss={history['train_loss'][-1]:.4f}  "
              f"val_loss={val_loss:.4f}  val_wf1={wf1:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def predict_tirex(model, X, batch_size=64, device="cuda"):
    """Inference with TiRexClassifier. X: (n, C, T) numpy array."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    all_preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        with torch.no_grad():
            all_preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(all_preds)
