import sys
import subprocess
import os

base_dir = r"c:\Users\abhay\Desktop\MRL TESTS"
results_dir = os.path.join(base_dir, "Results_and_Reports")
os.makedirs(results_dir, exist_ok=True)

v3_out = os.path.join(results_dir, "v3_detailed_results.txt")
v4_out = os.path.join(results_dir, "v4_detailed_results.txt")

print(f"Running V3 geometric analysis, saving stdout to {v3_out}...")
with open(v3_out, "w", encoding="utf-8") as f:
    subprocess.run(["python", "geometric_analysis.py"], stdout=f, cwd=os.path.join(base_dir, "Scripts"))

print(f"Running V4 geometric analysis, saving stdout to {v4_out}...")
with open(v4_out, "w", encoding="utf-8") as f:
    subprocess.run(["python", "geometric_analysis_v4_p1.py"], stdout=f, cwd=os.path.join(base_dir, "Scripts"))

print("Done generating detailed results.")
