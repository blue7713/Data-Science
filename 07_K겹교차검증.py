import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# 데이터 만들기
X = np.linspace(-5, 3, 100) # -5~3까지를 등간격으로 100등분
y = 0.1*(X-3)*(X+3)*(X+1)*(X+5) + np.random.normal(1, size=100)
plt.scatter(X,y)
plt.show()
print(X)
print(y)

# SKlearn 이용한 교차검증
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5) # K겹 교차검증 객체 생성

# 루프를 통해 어떻게 인덱스가 생성되는지 확인
iteration = 1
for train_idx, test_idx in kfold.split(X, y): # n_split : 5 이므로 100개를 5등분하여 train과 test로 나눔(4:1)
    print('='*10, iteration, '='*10)
    print('train_idx: ', train_idx)
    print('test_idx: ', test_idx)
    iteration += 1

# 루프를 통해 어떻게 인덱스가 생성되는지 확인
iteration = 1
for train_idx, test_idx in kfold.split(X, y):
    print('='*10, iteration, '='*10)
    print('train_X: ', X[train_idx]) # X에 대한 훈련데이터
    print('test_X: ', X[test_idx]) # X에 대한 테스트데이터
    print('train_y: ', y[train_idx]) # y에 대한 훈련데이터
    print('test_y: ', y[test_idx]) # y에 대한 테스트데이터
    iteration += 1

# Shuffle을 지정하여 무작위로 데이터 나누기
kfold = KFold(n_splits=5, shuffle=True)
iteration = 1
kfold.split(X, y)
for train_idx, test_idx in kfold.split(X, y):
    print('='*10, iteration, '='*10)
    print('train_idx: ', train_idx)
    print('test_idx: ', test_idx)
    iteration += 1

# K겹 교차검증을 통해 모델평가해보기
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None

degrees = range(1, 20) # 다항식의 범위 지정
kfold = KFold(n_splits=5, shuffle=True) # 데이터 쪼개기

# 하나의 모델
test_mse = [] # 5번 반복하면서 테스팅 데이터의 정확도 기록
train_mse = [] # 5번 반복하면서 훈련 데이터의 정확도 기록

# k겹 교차검증
for train_idx, test_idx in kfold.split(X, y):

    # k겹 교차검증을 위한 데이터 쪼개기
    X_train = X[train_idx].reshape(-1, 1)
    y_train = y[train_idx]
    X_test = X[test_idx].reshape(-1, 1)
    y_test = y[test_idx]

    # 차수를 번갈아가며 기록
    test_mse_temp = {}
    train_mse_temp = {}
    for i in degrees:

        reg = LinearRegression() # 선형회귀 객체 생성
        poly = PolynomialFeatures(degree=i, include_bias=False) # 다항식 변환 객체

        reg.fit(poly.fit_transform(X_train), y_train) # 학습

        y_pred_train = reg.predict(poly.fit_transform(X_train)) # 훈련데이터를 입력으로 하여 예측
        y_pred_test = reg.predict(poly.fit_transform(X_test)) # 테스팅데이터를 입력으로 하여 예측

        train_mse_temp[i] = mean_squared_error(y_train, y_pred_train) # 훈련데이터의 정확도 계산
        test_mse_temp[i] = mean_squared_error(y_test, y_pred_test) # 테스팅데이터의 정확도 계산

    train_mse.append(train_mse_temp)
    test_mse.append(test_mse_temp)

train_acc = pd.DataFrame(train_mse)
test_acc = pd.DataFrame(test_mse)
print(train_acc)
print(test_acc)        

train_acc_mean = train_acc.apply(np.mean, axis=0) # 행을 가로질러 평균(열 평균) 계산
test_acc_mean = test_acc.apply(np.mean, axis=0)
print(train_acc_mean)
print(test_acc_mean)

# 교차검증의 결과를 차수별로 그래프로 표시
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(degrees, test_acc_mean, label='test')
plt.plot(degrees, train_acc_mean, label='train')
plt.legend()
plt.show()