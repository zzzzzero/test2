import pandas as pd
import numpy as np


files = ['submission_b4_47.8.csv', '0.5340.csv', '0.5357.csv', '0.5379.csv']
weights = [1, 1, 1, 1]

results = np.zeros((100000, 5000))
for file, w in zip(files, weights):

    df = pd.read_csv(file, header=None).values
    # print(df)
    for i,(x, y) in enumerate(df):
        # print(x, y)
        results[i, y] += w
        # break

print(results[4])

submit = {
    'name': np.arange(100000).tolist(),
    'pred': np.argmax(results, axis=1).tolist()
    }

for k, v in submit.items():
    print(k, v)

df = pd.DataFrame(submit)
df.to_csv('vote.csv', header=False, index=False)



