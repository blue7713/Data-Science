# 회귀_자동차연비예측
import pandas as pd
pd.options.display.max_columns = None # 모든 열 출력
pd.options.display.width = None # 줄바꿈 하지않고 출력
pd.options.display.max_colwidth = None # 열 너비 제한 없이 출력
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
print(df)

# Pairplot 그리기
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()

# 누락치 등 전처리
print(df.isnull().sum())

# Categorical Variable 처리
df = pd.get_dummies(df, columns=['origin'])
print(df)

# X, y를 나누기
y = df['mpg']
print(y)
X = df.drop(columns=['mpg'])
print(X)

# 회귀분석 모듈 불러오기
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# 학습
reg.fit(X, y)

# 예측
y_pred = reg.predict(X)

# 학습된 계수 구하기
print(reg.coef_)
print(X.columns)

# 막대그래프를 통해 각 계수의 크기를 비교
plt.figure(figsize = (12, 8)) # 사이즈 정하기
plt.bar(X.columns, reg.coef_) # 막대그래프
plt.show()

# 데이터를 정규화하여 학습
X_norm = ((X - X.mean()) / X.std())
print(X_norm)

reg.fit(X_norm, y)

plt.figure(figsize = (12, 8))
plt.bar(X.columns, reg.coef_)
plt.show()

# 예측값 vs 실제값
y_pred = reg.predict(X_norm)
plt.scatter(y, y_pred) # 완벽한 예측 x
plt.show() 

# 비용계산
import numpy as np
np.sqrt(np.sum((y - y_pred)**2)/ y.count())