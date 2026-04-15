import json

path = r"c:\Users\abhay\Desktop\MRL TESTS\Results_and_Reports\bias_results_v4.json"
out  = r"c:\Users\abhay\Desktop\MRL TESTS\Results_and_Reports\bias_summary_v4.txt"
d = json.load(open(path))

lines = []
for m in ['standard', 'mrl_v4']:
    lines.append(f"\n=== {m.upper()} ===")
    lines.append(f"{'Dim':>5} | {'WEAT':>7} | {'DirBias':>7} | {'RIPA_m':>7} | {'RIPA_x':>7} | {'ECT':>7} | {'NBM':>7} | {'ClustP':>7}")
    lines.append("-" * 75)
    for dim in ['50','100','150','200','250','300']:
        v = d[m][dim]
        lines.append(f"{dim:>5} | {v['weat_effect_size']:>7.4f} | {v['direct_bias']:>7.4f} | {v['ripa_mean']:>7.4f} | {v['ripa_max']:>7.4f} | {v['ect_spearman']:>7.4f} | {v['nbm_mean']:>7.4f} | {v['cluster_purity']:>7.4f}")

for m in ['standard', 'mrl_v4']:
    lines.append(f"\n=== {m.upper()} Analogies (d=300) ===")
    for w, r, s in d[m]['300'].get('analogy_results', []):
        lines.append(f"  he:she :: {w:15} -> {r:15} (sim={s})")

text = "\n".join(lines)
print(text)
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print(f"\nSaved to {out}")
