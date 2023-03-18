# 데이터 불러오기
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/heart.csv")
print(df)

# feature 별 기본 통계량
print(df.describe())

## 누락치 제거
print(df.isnull().sum())
df = df.dropna(axis = 0)
print(df)

# 데이터의 종류
# numeric(실수형 값) : 키, 온도, 점수, 돈...
# categorical(범주형 값) : 성별, 브랜드, 고장여부, 질병여부...
# 변수 내의 서로 다른 값이 서로 다른 의미 -> 비교가 불가능
# dummy variable : one-hot encodding
# 성별 -> (남자, 여자)
# 성별_남자, 성별_여자

# 범주형 변수의 처리 : one-hot encodding
print(df.head())
c = ['Sex', 'ChestPain', 'Thal', 'AHD']
df2 = pd.get_dummies(df, columns=c)

# 변수의 목록 출력
print(df2.columns)

# 변수의 표준화
# 표준화 하기 전 : 변수간에 서로 다른 scale, unit
# 특히 거리 기반의 머신러닝 방법
# min-max scaling : 변수 내에서 가장 큰값 1, 가장 작은값을 0
# standardize(표준화) : 변수가 평균이 0, 표준편차가 1이 되도록 변환

# 표준화
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
print(df)

# standardize 
# 공식 -> 데이터 - 데이터의 평균 / (데이터의 표준편차)
df.mean() # 각 feature의 평균
df.std() # 열별 표준편차

# 표준화
df = (df - df.mean())/ df.std()
print(df)

# Min-max scaling
# 최대값이 1, 최소값이 0이 되도록 변환
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
print(df)

# 각 열의 최대값, 최솟값
df.max()
df.min()

df = (df-df.min())/(df.max() - df.min())
print(df)

# 변수의 표준화
# sklearn : 파이썬에서 표준화된 머신러닝 라이브러리
# sklearn에서 제공하는 데이터 표준화 모튤을 사용
# sklearn proprocessing의
# StandardScaler
# MinMaxScaler

# sklearn을 이용한 변환
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv")
print(df)

# 객체 생성
scaler = StandardScaler()
output = scaler.fit_transform(df) # numpy array형태로 변환시켜서 생성
print(output)
print(pd.DataFrame(output, columns=df.columns))

scaler = MinMaxScaler()
output = scaler.fit_transform(df)
print(output)
print(pd.DataFrame(output, columns= df.columns))

# Polynomial Term 추가하기
# 선형회귀 모형의 한계를 깨기 위한 방법
# 비선형적인 관계를 표현하는 방법 : 다차항 넣기, 변수간의 상호작용
# (회귀)X, X^2, X^3,...
# (분류)X1, X2 -> x1^2, x2^2, X1*X2
# sklearn에서 제공하는 함수로 손쉽게 해결 가능

# Polynomial Term 추가
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/classification_data.csv")
print(df)

# 독립변수 추리기
x = df[['X1', 'X2']]
print(x)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, include_bias=False) # 2차식으로 변환, 편향 없애기
output = poly.fit_transform(x)
print(output)
c = poly.get_feature_names_out(x.columns)
print(pd.DataFrame(output, columns=c))