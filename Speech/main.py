import os
import glob
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ────────────────────────────────────────────
# 1) Glob directly from the original Actor_* folders
# ────────────────────────────────────────────
ROOT = "/Users/dillonmaltese/Documents/git/AI/Speech/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"
all_paths = sorted(glob.glob(os.path.join(ROOT, "Actor_*", "*.wav")))

print(f"Found {len(all_paths)} files")  # should be 1440

emotion_map = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}

codes      = [os.path.basename(p).split("-")[2] for p in all_paths]
labels_str = [emotion_map[c] for c in codes]
label_names= sorted(set(labels_str))
label2id   = {lab:i for i,lab in enumerate(label_names)}
id2label   = {i:lab for lab,i in label2id.items()}
labels     = [label2id[lab] for lab in labels_str]

train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ────────────────────────────────────────────
# 2) Dataset with torchaudio + librosa fallback
# ────────────────────────────────────────────
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h", sampling_rate=16_000
)

class Wav2Vec2Dataset(Dataset):
    def __init__(self, paths, labels, processor):
        self.paths     = paths
        self.labels    = labels
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            wav, sr = torchaudio.load(p)
            wav = wav.squeeze().numpy()
        except Exception:
            # fallback to librosa
            wav, sr = librosa.load(p, sr=self.processor.feature_extractor.sampling_rate)
        # resample if needed
        if sr != self.processor.feature_extractor.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.processor.feature_extractor.sampling_rate)
            sr  = self.processor.feature_extractor.sampling_rate

        # tokenize (no padding here)
        inputs = self.processor(
            wav,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False
        )
        # squeeze batch dim
        inputs = {k: v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs

train_ds = Wav2Vec2Dataset(train_paths, train_labels, processor)
test_ds  = Wav2Vec2Dataset(test_paths,  test_labels,  processor)

# ────────────────────────────────────────────
# 3) Model & Trainer
# ────────────────────────────────────────────
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels   = len(label_names),
    label2id     = label2id,
    id2label     = id2label,
    problem_type = "single_label_classification"
)
# (optional) freeze the feature extractor
for param in model.wav2vec2.parameters():
    param.requires_grad = False

data_collator = DataCollatorWithPadding(processor, padding=True)

training_args = TrainingArguments(
    output_dir                  = "./wav2vec2-emotion",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    eval_strategy               = "epoch",      # new name for evaluation_strategy
    num_train_epochs            = 5,
    learning_rate               = 1e-4,
    weight_decay                = 0.01,
    logging_steps               = 50,
    save_steps                  = 200,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = test_ds,
    data_collator = data_collator,
    tokenizer     = processor,  # will become processing_class in v5
)

# ────────────────────────────────────────────
# 4) Train & Evaluate
# ────────────────────────────────────────────
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# ────────────────────────────────────────────
# 5) Detailed Metrics
# ────────────────────────────────────────────
y_true, y_pred = [], []
for batch in trainer.get_eval_dataloader():
    labels = batch.pop("labels")
    outputs = model(**{k:v.to(model.device) for k,v in batch.items()})
    preds = outputs.logits.argmax(dim=-1).cpu().tolist()
    y_pred.extend(preds)
    y_true.extend(labels.tolist())

print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.colorbar(label="Count")
plt.xticks(range(len(label_names)), label_names, rotation=45, ha="right")
plt.yticks(range(len(label_names)), label_names)
th = cm.max()/2
for i in range(len(label_names)):
    for j in range(len(label_names)):
        plt.text(j, i, cm[i,j],
                 ha="center", va="center",
                 color="white" if cm[i,j]>th else "black")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()