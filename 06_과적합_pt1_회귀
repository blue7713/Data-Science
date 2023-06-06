import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None

import numpy as np
np.random.seed(42) # 다른 컴퓨터에서도 난수의 발생 순서를 보장

# 정답 데이터 생성
X = np.random.uniform(size=50)*5 # 0~5 사이의 난수 50개 생성
y = 0.65*X**3 - 4*X**2 + 3 + np.random.normal(scale=0.6, size=50)
print(X)
print(y)

# 정답 데이터를 산점도로 출력
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.show()

# 데이터 쪼개기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True) 
# 전체 데이터를 6:4 (훈련 : 테스트)

# 데이터 확인
print(f'X_train: {X_train.shape}') # f 스트링, 문자열과 변수값을 합쳐서 사용
print(f'y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')

# 훈련데이터와 테스팅데이터 그려보기
import numpy as np
import matplotlib.pyplot as plt

# X,y 데이터를 X기준으로 모두 정렬

# Sort the y_train and y_test arrays based on the sorted X_train and X_test arrays
y_train = y_train[np.argsort(X_train)] # [3, 1, 2] -> argsort -> {1, 2, 0}(인덱스값)
y_test = y_test[np.argsort(X_test)]

# Sort the traning and test arrays
X_train = np.sort(X_train)
X_test = np.sort(X_test)

# plot the sorted training and test data
plt.scatter(X_train, y_train, c='blue', label='Training data') # label : 범례
plt.scatter(X_test, y_test, c='red', label='Test data')

# Add a Legend
plt.legend()

# show the plot
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# degree : 차수 
# graph : 그래프 그릴지 여부 결정

def plot_regressor(degree, graph=True):
    
    # 모델 생성
    reg = LinearRegression()

    # 데이터 변환
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_ = poly.fit_transform(X_train.reshape(-1, 1)) # 1차원 데이터를 2차원으로 reshape

    # 모델 학습
    reg.fit(X_train_, y_train)

    # 예측값
    y_pred_train = reg.predict(X_train_)

    # 그래프 그리기
    if graph:
        plt.scatter(X_train, y_train, c='blue', label='training')
        plt.scatter(X_test, y_test, c='red', label='testing')
        plt.plot(X_train, y_pred_train, color='blue')# 훈련데이터를 얼마나 잘 나타내는지 표현하기 위해 선형도 작성
        plt.legend()
        plt.show()

    # 트레이닝 데이터와 테스팅 데이터 각각의 MSE 계산
    X_test_ = poly.fit_transform(X_test.reshape(-1, 1))
    y_pred_test = reg.predict(X_test_)
    train_mse = mean_squared_error(y_train, y_pred_train) # 훈련 데이터의 MSE
    test_mse = mean_squared_error(y_test, y_pred_test) # 테스팅 데이터의 MSE
    print(f'training MSE : {train_mse}')
    print(f'testing MSE: {test_mse}')

    return train_mse, test_mse

# 1차 회귀
plot_regressor(1)

# 2차 회귀
plot_regressor(2)

# 3차 회귀
plot_regressor(3)

# 8차 회귀
plot_regressor(8)

# 15차 회귀
plot_regressor(15)

# 유연성을 변경시켜가며 mse 기록
train_mse = []
test_mse = []
degrees = range(1, 16)

for i in degrees:
    print("="*10 + str(i) + "="*10)
    train, test = plot_regressor(i, graph=False)
    train_mse.append(train)
    test_mse.append(test)

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_mse, label='train_mse')
plt.plot(degrees, test_mse, label='test_mse')
plt.legend()
plt.show()    