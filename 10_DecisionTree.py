import numpy as np
np.random.seed(42)

# Decision Tree(회귀문제)
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv')
print(df)

# 범주형 변수 바꾸기
# 이론상으로 Decision Tree는 범주형 변수를 바꿀 필요가 없다, 하지만 현재 sklearn에서는 범주형 변수를 직접적으로 다루는 기능이 추가되어 있지 않다. (다른 방법과의 통일을 위해서 그렇것으로 생각되어짐). 따라서 범주형 변수에 대하여 one-hot 인코딩을 진행한다
# 하지만 다른 방법론처럼 standard scaler를 쓸 필요는 없다

df = pd.get_dummies(df, columns=['origin'])
print(df)

# X와 y 구분하기
y = df['mpg']
X = df.drop(columns='mpg')
print(df.dtypes)

# Decision Tree의 시각화
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
reg = DecisionTreeRegressor(max_depth=2)
print(reg.fit(X,y))

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 8))
plot_tree(reg, feature_names=X.columns)
plt.show()

# 하이퍼패러미터 튜닝
from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits=5, shuffle=True) # 교차검증기
reg = DecisionTreeRegressor() # 학습모델 객체 생성
param_grid = {'max_depth' : [2, 3, 4, 5, 6, 7, 8, 9, 10]} # 하이퍼패러미터의 범위를 딕셔너리 형태로 설정
grid_search = GridSearchCV(estimator=reg,
                          param_grid=param_grid,
                          cv=kfold,
                          return_train_score=True,
                          scoring='neg_mean_squared_error')
result = grid_search.fit(X, y)
print(result)

import pandas as pd
df = pd.DataFrame(result.cv_results_)
print(df)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.plot(df['param_max_depth'], -df['mean_test_score'], label='test')
plt.plot(df['param_max_depth'], -df['mean_train_score'], label='train')
plt.legend()
plt.xlabel("max_depth")
plt.ylabel('mean_squared_error')
plt.show()

print(result.best_score_, result.best_params_)