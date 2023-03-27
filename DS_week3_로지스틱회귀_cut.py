# LogisticRegression
# 데이터셋 불러오기
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/classification_data.csv")
print(df)

# 데이터 그리기
import matplotlib.pyplot as plt

# 산점도 그리기
plt.scatter(df['X1'], df['X2'])
plt.show()

# scatter 패러미터 c: 각 observation의 색깔 지정
plt.scatter(df['X1'], df['X2'], c=df['y'])
plt.show()

# 로지스틱함수
import numpy as np
z = np.linspace(-10, 10, 100) # -10~10까지의 고르게 100등분
print(z)

# 로지스틱 함수 정의
def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

# 함수 작동 확인
# numpy array의 broadcasting(벡터를 함수에 넣으면 처리되어 다시 벡터로 나옴)
print(sigmoid(z))
plt.plot(z, sigmoid(z))
plt.show()

# 로지스틱회귀 학습
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

# 데이터 분할(독립, 종속)
X = df[['X1', 'X2']]
y = df['y']
reg.fit(X,y) # 학습

# 학습결과 확인(패러미터)
w = reg.coef_ # x의 패러미터
b = reg.intercept_ # 상수 패러미터(y절편)

# 학습된 패러미터로 계산한 확률 = sklearn에서 계산한 확률
# 학습된 모델에 학습에 사용된 데이터 각각을 넣어서 각각이 0또는 1로 분류될 확률을 계산
print(reg.predict_proba(X)) # [0으로 분류될 확률, 1로 분류될 확률]
print(reg.predict_proba(X)[:, 1]) # 학습에 사용된 데이터 각각이 1로 분류될 확률

# signoid(w*X + b) , X와 w는 사이즈가 다르므로 T로 w의 행과 열을 맞바꿔줌
print(sigmoid(np.dot(X, w.T) + b)) # 선형회귀로 예측되는 값

# 정확도 계산
y_pred = reg.predict(X) # 예측된 y값
print(y_pred)

# numpy array의 broadcasting
print(np.sum(y == y_pred)/y.count()) # numpy에서 sum은 True는 1, False는 0으로 간주, 89% 적중

# 의사결정의 경계선
# w1*X1 + w2*X2 + b = 0
# W2*X2 = -w1*X1 - b
# X2 = -(w1*X1 + b) / w2
x1 = np.linspace(df['X1'].min(), df['X1'].max(), 50)
x2 = (-1*(b + w[0][0]* x1)/w[0][1]) # X2 = -(w1*X1 + b) / w2
plt.scatter(df['X1'], df['X2'], c=df['y'])
plt.plot(x1, x2)
plt.xlim(df['X1'].min() - 1, df['X1'].max() + 1) # x축 범위 제한
plt.ylim(df['X2'].min() - 1, df['X2'].max() + 1) # y축 범위 제한
plt.show()