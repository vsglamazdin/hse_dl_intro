import os
import pandas as pd

root = os.path.join("data", "data")

folders = os.listdir(root)
data = []
for i in folders:
    data.append((i, len(os.listdir(os.path.join(root,i)))))
pd.DataFrame(data).to_csv(os.path.join("res.csv"), index = None)

print(data)

x =input()
