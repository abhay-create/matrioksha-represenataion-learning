"""Assemble mrl-v3-cells/*.py into mrl-v3.ipynb"""
import json, glob, os

cells_dir = r"C:\Users\abhay\Desktop\MRL TESTS\mrl-v3-cells"
out_path  = r"C:\Users\abhay\Desktop\MRL TESTS\mrl-v3.ipynb"

cell_files = sorted(glob.glob(os.path.join(cells_dir, "cell*.py")))
print(f"Found {len(cell_files)} cell files:")
for f in cell_files:
    print(f"  {os.path.basename(f)}")

cells = []
for fpath in cell_files:
    with open(fpath, "r", encoding="utf-8") as f:
        source = f.read()
    # Split into lines, each ending with \n (notebook format)
    lines = source.split("\n")
    source_list = [line + "\n" for line in lines[:-1]]  # all but last
    if lines[-1]:  # last line without trailing newline
        source_list.append(lines[-1])

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source_list,
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "dockerImageVersionId": 31287,
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✓ Created {out_path}")
print(f"  Size: {os.path.getsize(out_path)/1024:.1f} KB")
print(f"  Cells: {len(cells)}")
