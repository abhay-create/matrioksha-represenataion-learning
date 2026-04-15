import json
path = r"c:\Users\abhay\Desktop\MRL TESTS\Data_and_Embeddings\_v3_embeddings_W2V\Embeddings\mrl_bias_v4_embeddings\bias_results.json"
d = json.load(open(path))
for m in ['standard', 'mrl_v4']:
    print(f"\n{m.upper()}")
    for dim in ['50','100','150','200','250','300']:
        v = d[m][dim]
        print(f"d{dim} W{v['weat']:.3f} D{v['direct_bias']:.4f} Rm{v['ripa_mean']:.4f} Rx{v['ripa_max']:.4f} E{v['ect']:.4f} N{v['nbm']:.4f} C{v['cluster_purity']:.4f}")
for m in ['standard', 'mrl_v4']:
    print(f"\n{m.upper()} ANALOGIES")
    for w, r, s in d[m]['300'].get('analogies', []):
        print(f"  {w} -> {r}")
