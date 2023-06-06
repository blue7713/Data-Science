# 데이터 불러오기
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv(
    "https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
print(df)

# 전처리
# origin 변수 one-hot encodding
df = pd.get_dummies(df, columns=['origin'])
print(df)

# 독립, 종속변수 나누기
y = df['mpg']
X = df.drop(columns=['mpg'])
print(y)

# 선형회귀분석 실행
reg = LinearRegression()
reg.fit(X, y)

y_pred = reg.predict(X)  # numpy 배열
print(y_pred)

# MAE(mean absolute error)
print(np.mean(np.abs(y - y_pred)))  # MAE

print(mean_absolute_error(y, y_pred))

# MAPE(mean absolute percent error)
print(np.mean(np.abs(y - y_pred)/y))
print(mean_absolute_percentage_error(y, y_pred))

# y의 단위가 각 지표에 미치는 영향
z = y * 0.01  # 정답 데이터의 스케일을 100분에 1로 낮춤
z_pred = y_pred * 0.01  # 예측값이 스케일을 100분에 1로 낮춤

print("mse: ", mean_squared_error(z, z_pred))
print("r2_score: ", r2_score(z, z_pred))
print("mae : ", mean_absolute_error(z, z_pred))
print("mape : ", mean_absolute_percentage_error(z, z_pred))

# MAPE의 값이 이상하게 나오는 경우
z[0] = 0.0000000000000000001  # 전체 레이블의 정답이 0에 가까운 경우
print("mse: ", mean_squared_error(z, z_pred))
print("r2_score: ", r2_score(z, z_pred))
print("mae : ", mean_absolute_error(z, z_pred))
print("mape : ", mean_absolute_percentage_error(z, z_pred))
