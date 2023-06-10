import numpy as np
np.random.seed(42)

import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df=pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv')
print(df)

df = pd.get_dummies(df, columns=['origin'])
print(df)

# 데이터 X,y 분리하기
y=df['mpg']
X=df.drop(columns='mpg')

# 학습 준비
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
kfold = KFold(n_splits=5, shuffle=True)
reg = RandomForestRegressor()
param_grid = {'max_depth':[2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(estimator=reg, cv=kfold, param_grid=param_grid, scoring='neg_mean_squared_error', return_train_score=True)
result = grid_search.fit(X,y)
print(result)

# 하이퍼패러미터 변화에 따른 성능 시각화
import matplotlib.pyplot as plt
result_df = pd.DataFrame(result.cv_results_)
plt.figure(figsize=(8, 5,))
plt.plot(result_df['param_max_depth'], -result_df['mean_test_score'], label='test')
plt.plot(result_df['param_max_depth'], -result_df['mean_train_score'], label='train')
plt.legend()
plt.show()

## 성능 확인
print(result.best_score_, result.best_params_)

# 하이퍼패러미터 튜닝
kfold = KFold(n_splits=5, shuffle=True)
reg = RandomForestRegressor()
param_grid = {'max_depth':[2,5,10,30,50],
              'n_estimators': [50,100,200,500],
              'max_features': ['sqrt','log2']              
             }
grid_search = GridSearchCV(estimator=reg, cv=kfold, param_grid=param_grid, scoring='neg_mean_squared_error', return_train_score=True)
result = grid_search.fit(X,y)

print(result.best_score_, result.best_params_)

# 분류문제
## 데이터 불러오기
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/heart.csv')
df=df.dropna()
print(df)

## 원핫인코딩
df = pd.get_dummies(df, columns=['Sex','AHD','ChestPain','Thal','AHD'])
print(df)

## X와 y 구분
y = df['AHD_Yes']
X = df.drop(columns=['AHD_Yes','AHD_No'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV

## 학습준비
kfold = KFold(n_splits=5, shuffle=True)
clf = RandomForestClassifier()
param_grid = {'max_depth':[2,5,10,30,50],
              'n_estimators': [50,100,200,500],
              'max_features': ['sqrt','log2']              
             }
grid_search = GridSearchCV(estimator=clf, cv=kfold, param_grid=param_grid, scoring='accuracy', return_train_score=True)
result = grid_search.fit(X,y)

## 성능 확인
print(result.best_score_, result.best_params_)