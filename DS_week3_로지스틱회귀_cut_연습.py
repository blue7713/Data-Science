# 타이타닉 생존 예측
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/titanic_processed.csv")
print(df)

# 학습
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

# 데이터 분리(종속, 독립)
# 리스트 슬라이싱
# list[시작인덱스 : 종료인덱스] -> 시작 ~ 종료 -1
X = df[df.columns[:-1]] # 첫 컬럼부터 마지막 앞까지 컬럼
y = df['survived']

# 학습
reg.fit(X, y)

# 정확도 측정
y_pred = reg.predict(X)

import numpy as np
print(np.sum(y_pred == y)/y.count())

# 결과 해석
import matplotlib.pyplot as plt
plt.bar(X.columns, reg.coef_[0]) # reg.coef_는 2차원이므로 1차원으로 만들기 위함
plt.show()

plt.figure(figsize=(16, 8))
plt.bar(X.columns, reg.coef_[0])
plt.show()

# 데이터의 표준화(정규화)
# 평균 0, 표준편차 1 표준 정규분포
X_norm = (X - X.mean())/X.std()
reg.fit(X_norm, y)
plt.figure(figsize=(16, 8))
plt.bar(X.columns, reg.coef_[0])
plt.show() # 로지스틱회귀나 선형회귀는 정확도는 낮지만 얻어진 계수들로 어떤것들이 크게 영향을 주는지 유추할 수 있음 