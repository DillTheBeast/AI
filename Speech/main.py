# simple_melcnn.py

import os
import glob
import random
import numpy as np
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────
# 0) Fix random seed
# ────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ────────────────────────────────────
# 1) Collect files & labels
# ────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "all_wavs")
files = sorted(glob.glob(os.path.join(ROOT, "*.wav")))
assert files, "No .wav files found in all_wavs/"

emotion_map = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}
codes      = [os.path.basename(f).split("-")[2] for f in files]
labels_str = [emotion_map[c] for c in codes]
LABELS     = sorted(set(labels_str))
lab2idx    = {l:i for i,l in enumerate(LABELS)}
labels     = [lab2idx[l] for l in labels_str]

# stratified 80/20 split
train_files, test_files, y_train, y_test = train_test_split(
    files, labels,
    test_size   = 0.2,
    random_state=SEED,
    stratify    = labels
)

# ────────────────────────────────────
# 2) Mel-spectrogram transform
# ────────────────────────────────────
SR         = 16000
MAX_SEC    = 4.0
MAX_SAMPS  = int(SR * MAX_SEC)

mel_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=400, hop_length=160, n_mels=64
    ),
    torchaudio.transforms.AmplitudeToDB()
)

# ────────────────────────────────────
# 3) Dataset
# ────────────────────────────────────
class RAVDESSDataset(Dataset):
    def __init__(self, files, labels):
        self.files  = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        wav, sr = torchaudio.load(f)          # [channels, time]
        wav = wav.mean(0, keepdim=True)       # to mono
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        # pad or trim
        if wav.size(1) < MAX_SAMPS:
            wav = nn.functional.pad(wav, (0, MAX_SAMPS - wav.size(1)))
        else:
            wav = wav[:, :MAX_SAMPS]
        spec = mel_transform(wav)             # [1, n_mels, time]
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec, self.labels[idx]

train_ds = RAVDESSDataset(train_files, y_train)
test_ds  = RAVDESSDataset(test_files,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

# ────────────────────────────────────
# 4) Simple 2‑layer CNN
# ────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=len(LABELS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*(64//4)*(int(MAX_SAMPS/160)//4), 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SimpleCNN().to(device)
opt    = optim.Adam(model.parameters(), lr=1e-3)
loss_fn= nn.CrossEntropyLoss()

# ────────────────────────────────────
# 5) Train & evaluate
# ────────────────────────────────────
for epoch in range(1, 51):
    # train
    model.train()
    total_loss = 0
    for X,y in train_loader:
        X,y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss   = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()*X.size(0)
    train_loss = total_loss/len(train_ds)

    # eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for X,y in test_loader:
            X,y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            correct += (preds==y).sum().item()
    acc = correct/len(test_ds)
    print(f"Epoch {epoch:02d} | Train Loss {train_loss:.3f} | Val Acc {acc:.3f}")

# ────────────────────────────────────
# 6) Final report & confusion matrix
# ────────────────────────────────────
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X,y in test_loader:
        X = X.to(device)
        preds = model(X).argmax(1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

print(classification_report(y_true, y_pred, target_names=LABELS))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
plt.yticks(range(len(LABELS)), LABELS)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix to confusion_matrix.png")
