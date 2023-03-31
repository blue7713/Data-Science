# 회귀모형의 평가
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
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
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

# 학습된 회귀모형 평가
# 학습에 사용된 데이터 -> 학습된 모형 -> 예측값 = y_pred
# 학습에 사용된 데이터의 정답 -> y
# RMSE(rooted mean squared error)
# R2-score

# RMSE
y_pred = reg.predict(X) # numpy 배열
print(y_pred) 

# 실제 데이터의 정답
# pandas Series
# RMSE
import numpy as np
rmse = np.sqrt(np.sum(((y_pred - y)**2)/y.size))
print(rmse)

# R2_Score
y_mean = y.mean()
print(y_mean)

tss = np.sum((y - y_mean)**2)
print(tss)

rss = np.sum((y - y_pred)**2)
print(rss)

r2_Score = 1 - rss/tss
print(r2_Score)

# sklearn 패키지를 이용한 평가지표 구하기
from sklearn.metrics import mean_squared_error, r2_score
print(np.sqrt(mean_squared_error(y, y_pred)))
print(r2_score(y, y_pred))