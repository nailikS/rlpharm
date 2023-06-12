from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
# Read data
data_path = r"C:\Users\kilia\MASTER\rlpharm\data\approxCollection.csv"
df = read_csv(data_path)
df_test = df.sample(frac=0.2)

df = df.loc[~df.index.isin(df_test.index)]

X = df.iloc[:, 1:-4].values
y = df.iloc[:, -1].values#.reshape(-1, 1)

# define model
model = XGBRegressor()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.4f (%.4f)' % (scores.mean(), scores.std()))