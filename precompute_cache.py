import pandas as pd
import numpy as np
import pickle
from itertools import product

df = pd.read_csv('dashboard/df_clean.csv')
with open('dashboard/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('dashboard/le_dict.pkl', 'rb') as f:
    le_dict = pickle.load(f)

feature_cols = ['online_order','book_table','votes','cost','location','rest_type','cuisines','listing_type']
sample_costs = [200, 500, 800, 1000, 1500, 2000, 3000]
sample_votes = [50, 100, 200, 400, 600, 1000, 2000]
key_locations = df['location'].value_counts().head(20).index.tolist()
key_rts = ['Quick Bites', 'Casual Dining', 'Cafe', 'Delivery']
key_cuisines = ['North Indian', 'Chinese', 'South Indian', 'Fast Food', 'Italian']

rows = []
keys = []
for loc, rt, cu, online, booktable, cost, votes in product(
        key_locations, key_rts, key_cuisines, [0,1], [0,1], sample_costs, sample_votes):
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

X_cache = pd.DataFrame(rows, columns=feature_cols)
probs = model.predict_proba(X_cache)[:,1]
prediction_cache = {k: float(p) for k, p in zip(keys, probs)}

with open('dashboard/prediction_cache.pkl', 'wb') as f:
    pickle.dump(prediction_cache, f)

print(f'Done: {len(prediction_cache)} predictions saved')