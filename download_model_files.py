import os

files = {
    "image_paths.npy":     "YOUR_PATHS_FILE_ID",
    "image_features.npy":  "YOUR_FEATURES_FILE_ID",
    "faiss_index.bin":     "YOUR_FAISS_FILE_ID",
}

os.makedirs("models", exist_ok=True)

for fname, fid in files.items():
    if not os.path.exists(f"models/{fname}"):
        print(f"Downloading {fname}...")
        os.system(f"gdown --id {fid} -O models/{fname}")
    else:
        print(f"{fname} already exists — skipping.")
