import pandas as pd
import numpy as np
import pickle
from itertools import product
from tqdm import tqdm

df = pd.read_csv('dashboard/df_clean.csv')
with open('dashboard/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('dashboard/le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

feature_cols = ['online_order','book_table','votes','cost','location','rest_type','cuisines','listing_type']
sample_costs = list(range(100, 3001, 100))   # 30 points - keep this
sample_votes = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]  # 12 points instead of 41
key_locations = df['location'].value_counts().head(20).index.tolist()
key_rts = sorted(df['rest_type'].str.split(',').explode().str.strip().unique().tolist())
# Top 30 cuisines by count
key_cuisines = (df['cuisines'].str.split(',').explode().str.strip().value_counts().head(30).index.tolist())
print(f"Rest types: {len(key_rts)}")
print(f"Cuisines: {len(key_cuisines)}")
key_cuisines = ['North Indian', 'Chinese', 'South Indian', 'Fast Food', 'Italian']

all_combos = list(product(key_locations, key_rts, key_cuisines, [0,1], [0,1], sample_costs, sample_votes))
print(f"Total combinations: {len(all_combos)}")

rows = []
keys = []
for loc, rt, cu, online, booktable, cost, votes in tqdm(all_combos, desc="Building cache"):
    try:
        loc_enc = le_dict['location'].transform([loc])[0]
        all_rt = le_dict['rest_type'].classes_
        rt_match = next((r for r in all_rt if rt in r), all_rt[0])
        rt_enc = le_dict['rest_type'].transform([rt_match])[0]
        all_cu = le_dict['cuisines'].classes_
        cu_match = next((c for c in all_cu if cu in c), all_cu[0])
        cu_enc = le_dict['cuisines'].transform([cu_match])[0]
        lt_enc = le_dict['listing_type'].transform(['Dine-out'])[0]
        rows.append([online, booktable, votes, cost, loc_enc, rt_enc, cu_enc, lt_enc])
        keys.append((loc, rt, cu, online, booktable, cost, votes))
    except:
        pass

print(f"Running batch prediction on {len(rows)} rows...")
X_cache = pd.DataFrame(rows, columns=feature_cols)
probs = model.predict_proba(X_cache)[:,1]
prediction_cache = {k: float(p) for k, p in zip(keys, probs)}

with open('dashboard/prediction_cache.pkl', 'wb') as f:
    pickle.dump(prediction_cache, f)

print(f'Done: {len(prediction_cache)} predictions saved')