import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv('data/zomato.csv')
df.drop(columns=['url','address','phone','reviews_list','menu_item'], inplace=True)
df.drop_duplicates(inplace=True)
df['rate'] = pd.to_numeric(df['rate'].astype(str).str.replace('/5','').str.strip(), errors='coerce')
df['approx_cost(for two people)'] = pd.to_numeric(
    df['approx_cost(for two people)'].astype(str).str.replace(',','').str.strip(), errors='coerce')
df.rename(columns={'approx_cost(for two people)':'cost',
                   'listed_in(type)':'listing_type',
                   'listed_in(city)':'listed_city'}, inplace=True)
df['online_order'] = (df['online_order']=='Yes').astype(int)
df['book_table'] = (df['book_table']=='Yes').astype(int)
df.dropna(subset=['rate','cost','location','cuisines','rest_type'], inplace=True)
df['high_performer'] = ((df['rate']>=4.0) & (df['votes']>=200)).astype(int)

df_enc = df.copy()
le_dict = {}
for col in ['location','rest_type','cuisines','listing_type','listed_city']:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    le_dict[col] = le

feature_cols = ['online_order','book_table','votes','cost',
                'location','rest_type','cuisines','listing_type']
X = df_enc[feature_cols]
y = df_enc['high_performer']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale_pos = (y==0).sum()/(y==1).sum()

model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
model.fit(X_tr, y_tr, verbose=True)

with open('dashboard/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('dashboard/le_dict.pkl', 'wb') as f:
    pickle.dump(le_dict, f)

df.to_csv('dashboard/df_clean.csv', index=False)
print("All files saved successfully!")