# 데이터 전처리
# 누란된 데이터
# 결측치, 잘못된 값
# 중복되는 값
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/datacleaning.csv')
print(df)

# 데이터프레임 내부에서 NaN -> 값이 없음
# 누락된 데이터 확인하기
print(df.isnull())

# 연쇄적 명령
# 결과 <- df.isnull()
# 결과.sum(), True = 1, False = 0
print(df.isnull().sum())

# 결측치가 들어있는 행 제거하기 (axis=0)
df2 = df.dropna(axis=0) # df 자체 값이 바뀌는 것이 아님
print(df2)

# 결측치가 들어있는 열 제거하기 (axis=1)
df3 = df.dropna(axis=1)
print(df3)

# 결측치 메꾸기
df4 = df.fillna(130) # 결측치를 130이라는 값으로 메꾼다.
print(df4)

df['Pulse'] = df['Pulse'].fillna(130)
print(df)

df['Calories'] = df['Calories'].fillna(300)
print(df)

# 평균값을 이용한 결측치 채우기
df = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/datacleaning.csv')
print(df)

df['Pulse'] = df['Pulse'].fillna(df['Pulse'].mean()) # Pulse 열의 평균(결측치 제외)
df['Calories'] = df['Calories'].fillna(df['Calories'].mean()) # Calories 열의 평균(결측치 제외)
print(df)

# 한번에 실행
# df.mean() 각 열의 평균값
# df.fillna()
df.fillna(df.mean())
print(df)

# 중복된 값 제거
print(df.drop_duplicates()) # 기본옵션, 첫번째 값만 남기고 제거

print(df.drop_duplicates(keep= 'last')) # 마지막 것만 남기고 제거 