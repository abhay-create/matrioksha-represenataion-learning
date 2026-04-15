import json

path = r"c:\Users\abhay\Desktop\MRL TESTS\Results_and_Reports\bias_results_v4.json"
d = json.load(open(path))

# Print Standard table
for m in ['standard', 'mrl_v4']:
    print(f"\n{m.upper()}")
    for dim in ['50','100','150','200','250','300']:
        v = d[m][dim]
        w = v['weat_effect_size']
        db = v['direct_bias']
        rm = v['ripa_mean']
        rx = v['ripa_max']
        e = v['ect_spearman']
        n = v['nbm_mean']
        c = v['cluster_purity']
        print(f"d{dim} W{w:.3f} D{db:.4f} Rm{rm:.4f} Rx{rx:.4f} E{e:.4f} N{n:.4f} C{c:.4f}")

# Print analogies
for m in ['standard', 'mrl_v4']:
    print(f"\n{m.upper()} ANALOGIES")
    for w, r, s in d[m]['300'].get('analogy_results', []):
        print(f"  {w} -> {r}")
