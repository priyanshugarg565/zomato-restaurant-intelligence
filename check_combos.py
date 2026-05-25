import pandas as pd

df = pd.read_csv('dashboard/df_clean.csv')
key_rts = sorted(df['rest_type'].str.split(',').explode().str.strip().unique().tolist())
key_cuisines = df['cuisines'].str.split(',').explode().str.strip().value_counts().head(30).index.tolist()
sample_costs = list(range(100, 3001, 100))
sample_votes = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]
key_locations = df['location'].value_counts().head(20).index.tolist()

total = len(key_locations) * len(key_rts) * len(key_cuisines) * 2 * 2 * len(sample_costs) * len(sample_votes)
print(f"Rest types: {len(key_rts)}")
print(f"Cuisines: {len(key_cuisines)}")
print(f"Total: {total:,}")