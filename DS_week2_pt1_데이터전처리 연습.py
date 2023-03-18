# 타이타닉 데이터 전처리
import pandas as pd
from sklearn.datasets import fetch_openml
pd.options.display.max_columns = None # 모든 열 출력
pd.options.display.width = None # 자동으로 줄바꿈하지 않고 출력
pd.options.display.max_colwidth = None # 열 너비 제한 없이 출력.
X, y = fetch_openml("titanic", version = 1, as_frame=True, return_X_y=True)

print(X)
print(y)

# 각 열별로 고유한 값의 갯수가 몇개인지 
print(X.nunique())
print(X.isnull().sum())

colums = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = X[colums]
print(X)

# 누락 데이터 처리
print(X.isnull().sum())
X = X.dropna(axis=0)
print(X)

# 범주형 데이터 처리
X = pd.get_dummies(X, columns=['pclass', 'sex', 'embarked'])
print(X)

X = X.drop(columns='sex_male') # 성별은 한개만 있어도 됨
print(X)

# y값도 X에 맞추어 추려내기
print(X.index)
df = X
df['survived'] = y[X.index] # y중 X의 인덱스에 해당하는 것만 출력
print(df)