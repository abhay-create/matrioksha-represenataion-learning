import os
import shutil
import glob

base = r"c:\Users\abhay\Desktop\MRL TESTS"

folders = ["Scripts", "Notebooks", "Results_and_Reports", "Data_and_Embeddings"]
for f in folders:
    os.makedirs(os.path.join(base, f), exist_ok=True)

moves = {
    "Scripts": ["*.py"],
    "Notebooks": ["*.ipynb", "mrl-v3-cells"],
    "Results_and_Reports": ["*.txt", "*.json", "*.pdf", "geometry_plots"],
    "Data_and_Embeddings": ["*.zip", "*.csv", "*_embeddings*"]
}

for folder, patterns in moves.items():
    dest_dir = os.path.join(base, folder)
    for pat in patterns:
        for match in glob.glob(os.path.join(base, pat)):
            name = os.path.basename(match)
            if name in folders: continue
            if name == "cleanup.py": continue
            try:
                shutil.move(match, dest_dir)
                print(f"Moved {name} to {folder}")
            except Exception as e:
                print(f"Failed to move {name}: {e}")
