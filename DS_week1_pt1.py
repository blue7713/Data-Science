# pandas : csv 파일 분석, 열 선택, 다른 데이터로의 변환, 연산, 그래프 라이브러리
# pandas 열 선택
# pandas csv(엑셀형태의 파일,Comma seperated value) 불러오기
# pandas 연산
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/heart.csv')
print(df)

# 데이터프레임 기능
print(df.describe()) # 데이터프레임의 기초 통계량 계산(열별)
print(df.head()) # 첫 다섯줄
print(df.tail()) # 마지막 다섯줄
print(df.T.head()) # T는 행과열이 바뀜

# 열 선택하기
# df['칼럼명']
print(df['Age']) # 전체 데이터 중 Age 열만 고름
x = df['Sex']
print(x)

# 복수의 열 선택
# df[['칼럼명1', '칼럼명2']]
# 리스트 : 여러개의 값을 묶어서 하나의 값으로 나타냄
# 데이터프레임 안에 리스트 넣기
x = df[['Age', 'Sex', 'AHD']]
print(x)

# 열선택을 프로그래밍으로
colums = df.columns # 데이터 변수명들을 리스트(인덱스) 형태로 반환
print(colums)
print(colums[0:5]) # 0번째부터 4번째까지
print(df[colums[:5]])

# 리스트 컴프리헨션
# colums 안에 있는 각각의 특성중 대문자 C로 시작하는 것만 골라서 새로운 리스트로 추려내라
print(df[[i for i in colums if i.startswith('C')]])

# 연습문제
# 지금까지 했던 작업을 auto.csv 파일을 불러들여서 반복하시오
df1 = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv')
print(df1)

print(df1.describe())
print(df1.head())
print(df1.tail())
print(df1.T.head())

print(df1['cylinders'])
print(df1[['cylinders', 'displacement', 'horsepower']])

colum1 = df1.columns
print(colum1)
print(df1[[i for i in colum1 if i.startswith('h')]])

# 그래프 그리기(Matplotlib)
# line plot(선형 그래프), Scatter plot(산점도), Histogram, 그 외
import matplotlib.pyplot as plt

# 선형그래프(lineplot)
x = [1, 7, 5, 2, 5]
plt.plot(x)
plt.show()

# 그래프 겹쳐그리기
x = [1, 2, 3, 4, 5]
x2 = [2, 2, 3, 3, 4]
plt.plot(x)
plt.plot(x2)
plt.show()

plt.xlabel("x")
plt.ylabel("value")
plt.title("line plot example")
plt.show()

# 산점도(scatter) : 두 데이터 간의 경향성(상관성) 파악(2차원)
age = df['Age']
bp = df['RestBP']

# 데이터프레임의 칼럼이 바로 matplotlib의 입력값으로 사용가능
plt.scatter(age, bp)
plt.show()

plt.scatter(df['Sex'], bp)
plt.show()

# 히스토그램(histogram) : 단일 특성의 데이터 분포 파악(1차원)
plt.hist(df['Age'])
plt.show()

plt.hist(df['Age'], bins = 20) # 20개의 구간으로 그림
plt.show()

# 연습문제 
# 지금까지 했던 작업을 auto.csv 파일을 불러들여서 반복하시오
age1 = df1['cylinders']
bp1 = df1['displacement']

plt.scatter(df1['cylinders'], bp1)
plt.show()

plt.hist(df1['displacement'])
plt.show()

plt.hist(df1['displacement'], bins = 10)
plt.show()