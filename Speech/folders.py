import os
import glob
import shutil

# 1) Where your 24 Actor_* folders live
ROOT = "/Users/dillonmaltese/Documents/git/AI/Speech/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1"

# 2) Grab every .wav under those Actor_* subfolders
all_paths = sorted(glob.glob(os.path.join(ROOT, "Actor_*", "*.wav")))

# 3) Where you want to copy them all into
DEST = "/Users/dillonmaltese/Documents/git/AI/Speech/all_wavs"
os.makedirs(DEST, exist_ok=True)

# 4) Copy each file over
for src_path in all_paths:
    fname = os.path.basename(src_path)
    dst_path = os.path.join(DEST, fname)
    shutil.copy2(src_path, dst_path)

print(f"Copied {len(all_paths)} files into {DEST}")