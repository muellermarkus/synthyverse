import torch
import pandas as pd
from synthyverse.generators import TabbyFlowGenerator


# sample data for testing
x_num = torch.randn(1000, 2)
x_bin = torch.randint(0, 2, (1000, 2))
x_cat = torch.randint(0, 8, (1000, 2))
x_cat = torch.column_stack((x_bin, x_cat))
categories = [2, 2, 8]

df = pd.DataFrame(torch.column_stack((x_num, x_cat)), columns=["x1", "x2", "x3", "x4", "x5", "x6"])
discrete_features=["x3", "x4", "x5", "x6"]
df["x3"] = df["x3"].astype("category")
df["x4"] = df["x4"].astype("category")
df["x5"] = df["x5"].astype("category")
df["x6"] = df["x6"].astype("category")
discrete_feature_indices = [df.columns.get_loc(c) for c in discrete_features]


gen = TabbyFlowGenerator(epochs=4500)
gen.fit(df, discrete_features=discrete_features)
out = gen.generate(100)
print(f"TRUE data: {df.head()}")
print(f"SYNTH data: {out.head()}")