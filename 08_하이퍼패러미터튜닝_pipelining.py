import numpy as np
import matplotlib.pyplot as plt
random_state = np.random.seed(42)

# 데이터 만들기
X = np.linspace(-5, 3, 100) # -5~3까지의 수를 등간격으로 100개 생성
y = 0.1*(X-3)*(X+3)*(X+1)*(X+5) + np.random.normal(1, size=100)
plt.scatter(X, y)
plt.show()

# 파이프라이닝 : 머신러닝 과정을 한번에 쭉

# 다항식을 추가한회귀를 하기 위해 필요한 과정
# 0. 입력 X.reshape(-1,1)
# 1. PolynomialFeatures로 변환
# 2. LinearRegression으로 학습

# 파이프라이닝이란 머신러닝하는 과정에서 여러개의 객체를 순서대로 실행해야할 경우, 그 과정을 하나로 묶어서 관리할 수 있는 객체

# 만드는 방법

# 1. 파이프라인 객체 만들기
# 2. 파이프라인 안에 순서대로 머신러닝에 사용될 객체 넣기
# 3. fit으로 학습

# 다항식이 들어간 회귀모형을 파이프라이닝으로 정의
from sklearn.pipeline import make_pipeline # 파이프라인 만들기
from sklearn.preprocessing import PolynomialFeatures # 다항식이 추가되도록 데이터 변환
from sklearn.linear_model import LinearRegression # 선형회귀분석
from sklearn.model_selection import KFold, cross_validate # 교차검증용 객체

# 파이프라이닝에 들어갈 재료
poly = PolynomialFeatures(degree=2)
reg = LinearRegression()

# 파이프라인이 없을 때
# X_ = poly.fit_transform(X.reshape(-1, 1))
# reg.fit(X_, y)

# 파이프라인 만들기
pipe = make_pipeline(poly, reg) # 두 가지 스텝을 동시에 수행

# 파이프라인에 넣어서 학습
# X: 첫 번째 단계에서 들어가야할 데이터를 넣어준다
# y: 타겟변수(레이블)을 그래도 넣어준다
print(pipe.fit(X.reshape(-1, 1), y)) # 학습

# 파이프라이닝과 CrossValidate 같이 사용

# 파이프라인에 들어갈 재료 만들기
poly = PolynomialFeatures(degree=2)
reg = LinearRegression()

# 파이프라인 만들기
pipe = make_pipeline(poly, reg) # 두 가지 스텝을 동시에 수행

# 교차검증 객체
kfold = KFold(n_splits=5, shuffle=True) # 데이터 쪼갤 방식
metrics = ['neg_mean_absolute_error','neg_mean_squared_error','r2'] #기록할 메트릭

# 교차검증 수행
result = cross_validate(X=X.reshape(-1, 1), y=y,
                        estimator=pipe, # 파이프라인을 estimator로 두었음
                        scoring=metrics,
                        cv=kfold,
                        return_train_score=True)

# corss_validate의 결과는 pandas로 변환하면 보기 좋다

# cross_validate의 결과물 : {키 : [리스트]}
# pandas의 DataFrame으로 변환 가능
print(result)

# 결과를 데이터프레임으로 변환
import pandas as pd
df = pd.DataFrame(result)
print(df)

# 각각의 trial에 대한 평균값 계산
print(df.mean())

# 최적의 하이퍼패러미터 한번에 찾기 : GridSearchCV
# 1. 파이프라인 재료 만들기
# 2. Pipeline을 이용해 파이프라인 만들기 (Pileline 함수 사용)
# 3. 하이퍼패러미터 범주지정
# 4. 교차검증 객체 지정
# 5. GridSearchCV를 이용해서 하이퍼패러미터 튜닝 실행
# 6. 결과분석 및 최적 하이퍼패러미터 도출

# GridSearchCV를 이용한 예시
# 하이퍼패러미터1: 다항식 1차식부터 19차식 까지 변경
# 파이프라인을 사용함
# 기존의 루프를 사용한 하이퍼패러미터 탐색방법에서 GridSearchCV라는 방법을 이용한 자동화

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np

# 1) 파이프라인 재료 만들기
poly = PolynomialFeatures() # degree는 잡지 않는다.
reg = LinearRegression()

# 2) Pipline을 이용한 파이프라인 만들기
# pipe = make_pipeline(poly, reg)
pipe = Pipeline([('poly', poly), ('reg', reg)]) # 이름을 붙일 때는 튜플형식('이름', estimator객체)의 리스트로 순서대로 구현한다.

# 3) 하이퍼패러미터의 범주 지정
degrees = range(1, 20) # 1~19까지의 리스트
param_grid = {'poly__degree' : degrees} # 탐색할 하이퍼패러미터의 범주를 딕셔너리 형태로 지정(키 : 파이프라인스탭이름__패러미터, 밸류: 리스트)

# 4) 교차검증 객체 지정
kfold = KFold(n_splits=5, shuffle=True) # 데이터 쪼갤 방식

# 5) GridSearchCV를 이용하여 하이퍼패러미터 튜닝 자동화
grid_search = GridSearchCV(
    pipe, # 파이프라인
    param_grid, # 탐색할 패러미터 범주
    cv=kfold, # 교차검증 객체
    scoring='neg_mean_squared_error', # 스코어
    return_train_score=True # 훈련데이터 점수 반환 여부
)

# GridSearchCV 학습
print(grid_search.fit(X.reshape(-1, 1), y))

print(grid_search.cv_results_)

# GridSearchCV의 결과 분석
df = pd.DataFrame(grid_search.cv_results_)
print(df)

print(df.columns)

plt.plot(df['param_poly__degree'],-1*df['mean_train_score'])
plt.plot(df['param_poly__degree'],-1*df['mean_test_score'])
plt.show()

# 최적의 하이퍼패러미터
# print the best hyperparameters and mean score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", -1*grid_search.best_score_)

# 최적의 하이퍼패러미터로 학습된 모형을 사용
best_model = grid_search.best_estimator_
print(best_model)
# best_model.predict()

# 두 개의 하이퍼패러미터를 이용한 예시
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np

# 1) 파이프라인 재료 만들기
poly = PolynomialFeatures() #degree는 잡지 않는다.
reg = Ridge()

# 2) 파이프라인 만들기
pipe = Pipeline(steps=[('poly', PolynomialFeatures()), ('ridge', Ridge())])

# 3) 하이퍼패러미터 범위 지정
degrees = range(1,20) # poly의 degree에 대한 범위
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10] # ridge의 alpha에 대한 범위
param_grid = {'poly__degree': degrees,
              'ridge__alpha': alphas}

# 4) 교차검증 객체 만들기
kfold = KFold(n_splits=5, shuffle=True) #데이터 쪼갤 방식

# 5) GridSearchCV 만들기
grid_search = GridSearchCV(pipe, param_grid, cv=kfold, scoring='neg_mean_squared_error', return_train_score=True)

# 탐색시작
grid_search.fit(X.reshape(-1,1), y)

pd.DataFrame(grid_search.cv_results_)

# Print the best hyperparameters and mean score
print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", -1*grid_search.best_score_)