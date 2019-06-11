# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.columns)
new_columns = []
for column in df.columns:
    new_columns.append(column.lower().replace(' ','_'))
df.columns = new_columns
print(df.columns)
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
print(df.columns)
df[['established_date','acquired_date']] = df[['established_date','acquired_date']].apply(pd.to_datetime)

y = df['2016_deposits']
X = df.drop('2016_deposits', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 3) 
# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
for dataframe in [X_train, X_val]:
    for col_name in time_col:
        new_col_name = "since_"+col_name
        dataframe[new_col_name] = pd.datetime.now() - dataframe[col_name]
        dataframe[new_col_name] = dataframe[new_col_name].apply(lambda x: float(x.days)/365)
        dataframe.drop(col_name, axis=1, inplace=True)
print(X_train.shape)
print(X_val.shape)
print(X_val.columns)
print(X_train['since_established_date'][100])
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()
print(cat)
# Code starts here
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

le = LabelEncoder()
for df in [X_train, X_val]:
    for col in cat:
        df[col] = le.fit_transform(df[col])

X_train_temp = pd.get_dummies(data = X_train, columns = cat)
X_val_temp = pd.get_dummies(data = X_val, columns = cat)

print(X_train_temp.head())
# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt = DecisionTreeRegressor(random_state = 5)
dt.fit(X_train, y_train)

accuracy = dt.score(X_val, y_val)

y_pred = dt.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print(rmse)


# --------------
from xgboost import XGBRegressor


# Code starts here
xgb = XGBRegressor(max_depth=50, learning_rate=0.83, n_estimators=100)

xgb.fit(X_train, y_train)
accuracy = xgb.score(X_val, y_val)

y_pred = xgb.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_pred, y_val))

print(rmse)
# Code ends here


