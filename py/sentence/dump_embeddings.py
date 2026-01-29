import pandas as pd
import numpy as np

df = pd.read_csv("embeddings/test_rest_int8.csv")
features = [f"q{i}" for i in range(384)]

# 隨機挑一筆
#row = df.sample(1).iloc[0]


# fixed
row = df.iloc[2658]
text = row["text"]
emb = row[features].to_numpy(dtype="int8")
label = int(row["label"])

with open("random_embedding.h", "w") as f:
    f.write("#include <stdint.h>\n#include <stdbool.h>\n\n")
    f.write(f"// {text}\n")
    f.write("static const int8_t random_embedding[384] = {")
    f.write(",".join(map(str, emb)))
    f.write("};\n\n")
    f.write(f"static const bool valid_embedding = {'true' if label==1 else 'false'};\n")

with open("random_embedding_gemmini.h", "w") as f:
    f.write("#include <stdint.h>\n#include <stdbool.h>\n\n")
    f.write("#define BATCH_SIZE 1\n")
    f.write("#define IN_ROW_DIM 1\n")
    f.write("#define IN_COL_DIM 384\n")
    f.write("#define IN_CHANNELS 1\n\n")
    f.write(f"// {text}\n")
    f.write(
        "static const elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS] = "
    )
    f.write("{")
    f.write("{{")
    f.write("{")
    f.write(",".join("{" + str(v) + "}" for v in emb))
    f.write("}")
    f.write("}}")
    f.write("};\n\n")
    f.write(f"static const bool valid_embedding = {'true' if label==1 else 'false'};\n")
