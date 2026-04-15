import os
path = r"c:\Users\abhay\Desktop\MRL TESTS\Data_and_Embeddings\_v3_embeddings_W2V\Embeddings\mrl_bias_v4_embeddings\deep_bias_results.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines:
    print(line.rstrip())
