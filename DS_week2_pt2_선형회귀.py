# 선형회귀
# 가상 데이터 만들기
import numpy as np

# 맞추어야 하는 정답
true_w = 2
true_b = -3

# 평균이 0이고 표준편차가 1인 정규분포로부터
# 50개의 데이터를 무작위로 추출
X = np.random.normal(1, size=50)
print(X) 

# y는 다음과 같은 관계식에 의해 얻어졌다.(가정)
y = true_w * X + true_b # y = 2X - 3
print(y)

import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.show()

# noise
noise = np.random.normal(0.1, size=50)
print(noise)

y = y + noise # 실제 데이터들의 분포

plt.scatter(X, y)
plt.show()

# 선형회귀 학습하기
# sklearn
# 파이썬의 머신러닝 라이브러리(거의 표준처럼 사용)
# from sklearn.linear_model imoirt LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# linear regression의 fit 함수의 X 인자는 2차원 벡터를 요구함
# 현재 X는 1차원 이였으므로, reshape를 이용해서 2차원으로 바꾸어줌
X = X.reshape(-1, 1) # X를 2차원으로 만듬
reg.fit(X, y) # 선형회귀 모형 학습, X는 2차원 벡터여야 함

print(reg.coef_) # 예측한 X의 파라미터
print(reg.intercept_) # 예측한 y의 파라미터

# 학습된 모형을 이용한 예측
y_pred = reg.predict(X)
print(y_pred)

# 그래프 겹쳐그리기
plt.scatter(X,y)
plt.scatter(X,y_pred)
plt.show()

# 새로운 데이터로 예측
x2 = np.linspace(-3, 3, 50) #-3 ~ 3까지 50개의 숫자를 등간격으로 만듬
print(x2)

y2_pred = reg.predict(x2.reshape(-1, 1))
print(y2_pred)

plt.scatter(X, y)
plt.plot(x2, y2_pred)
plt.show()