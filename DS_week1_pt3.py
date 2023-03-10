# 범주형 데이터의 처리 
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/heart.csv")
print(df)

# get_dummies를 이용한 one-hot encodding
# 원핫인코딩할 변수선정
categorical_features = ['Sex', 'ChestPain', 'Thal', 'AHD']
df2 = pd.get_dummies(df, columns=categorical_features)
print(df2)

# 누락치 제거
df3 = df2.dropna(axis=0)
print(df3)