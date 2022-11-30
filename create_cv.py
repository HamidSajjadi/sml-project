import pandas as pd
from sklearn.model_selection import KFold
import os

df = pd.read_csv('./data/train.csv')
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
result = next(kfold.split(df), None)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cv')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
for idx, fold in enumerate(kfold.split(df)):
    cv_dir = os.path.join(data_dir, f"fold_{idx + 1}")
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)
    train_df: pd.DataFrame = df.iloc[fold[0]]
    test_fold: pd.DataFrame = df.iloc[fold[1]]
    train_df.to_csv(os.path.join(cv_dir, "train.csv"), index=False)
    test_fold.to_csv(os.path.join(cv_dir, "eval.csv"), index=False)


