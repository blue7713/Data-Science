# KNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None

# Generating Artificial Data
df = pd.read_csv('https://raw.githubusercontent.com/tonyfischetti/InteractiveLogisticRegression/master/moons.csv')
print(df)

plt.scatter(df['X1'], df['X2'], c=df['Y'])
plt.show()

# KNN by hands
# KNN을 직접 코딩으로 구현해 볼 것이다
# 알고리즘: 주어진 데이터 x에 대하여 x와 가장 가까운 k개의 이웃을 구하고, 이들의 결과를 평균내어 예측
# 이를 위해 데이터프레임을 numpy array 형태로 변환하고 X라고 명칭을 붙이겠다.
X = np.array(df[['X1', 'X2']])
y = np.array(df['Y'])

print(X.shape, y.shape)

# 다음과 같은 데이터가 들어왔을때 어떻게 예측하는지 보고싶다.
x = np.array([1, -1]) # 새로운 데이터셋 (X1, X2) = (1, -1)

# 유클리디안 거리
a = np.array([3,3])
b = np.array([-2,-1.5])
print(np.sqrt(np.sum((a-b)**2)))

# x 데이터와 모든 데이터 사이의 거리를 구해보자.
# numpy 계산의 브로드캐스팅 규칙을 사용하여 모든 점과 첫번째 데이터 사이의 거리를 쉽게 구할 수 있다.
# result의 결과값을 확인해보자

result = np.sqrt(np.sum((x-X)**2, axis=1)) # (1, -1)과 X사이의 거리가 순서대로 적혀있음
print(result)

# numpy의 argsort를 이용하면 각 데이터를 오름차순으로 정렬했을 때 각 데이터의 순위가 어떻게 되는지를 반환한다.
# 자기 자신을 제외하고 나와 가장 가까운 k개의 데이터는 다음과 같이 구할 수 있을 것이다

# argsort 예시
# 리스트안에 숫자를 정렬한다고 했을때 인덱스 기준으로 어떻게 나와야 되냐
k = np.array([1,3,2,100,20])
print(np.argsort(k))
print(k[np.argsort(k)])

# [-1,1]과 모든 데이터 사이의 거리의 argsort -> order
order = result.argsort()
print(order)

# 가까운 순위가 5보다 작은것들
print(X[order[:5]])

# 위 펜시 인덱싱을 이용하여 나와 가장 가까운 5개 데이터의 y값을 다음과 같이 추려보자
print(y[order[:5]].mean())
print(y[order[:100]].mean())

# 지금까지 했던 과정을 함수로 정의해보자

# 8_2 KNN using SKLEARN
df = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/auto.csv')
print(df)

# 데이터 나누기
df = pd.get_dummies(df, columns=['origin'])
print(df)
y = df['mpg']
X = df.drop(columns=['mpg'])

# 데이터 정규화
X_norm = (X-X.mean())/X.std()
print(X_norm)

# KNN 학습
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate

kfold = KFold(n_splits=5, shuffle=True)
train_mse = []
test_mse = []
for i in range(2, 30):
    knn = KNeighborsRegressor(n_neighbors=i)
    result = cross_validate(X = X_norm, y=y,
                            cv=kfold,
                            scoring=['r2', 'neg_mean_squared_error'],
                            estimator=knn,
                            return_train_score=True)
    print(i, result['test_r2'].mean(), result['test_neg_mean_squared_error'].mean())
    train_mse.append(-1*result['train_neg_mean_squared_error'].mean())
    test_mse.append(-1*result['test_neg_mean_squared_error'].mean())

# 하이퍼 패러미터의 영향
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(train_mse, label='train')
plt.plot(test_mse, label='test')
plt.legend()
plt.show()

# KNN using GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
import warnings

# 경고문구 없애기
warnings.filterwarnings('ignore')

# 파이프라인 재료 만들기
scaler = StandardScaler() # 데이터 스케일링
knn = KNeighborsRegressor() # KNN Regressor

# 파이프라인 만들기
pipe = Pipeline([('scaler', scaler), ('knn', knn)])

# 하이퍼패러미터 범주 지정
degree = range(1, 19)
param_grid = {'knn__n_neighbors' : degree}

# 교차검증 객체 지정
kfold = KFold(n_splits=5, shuffle=True)

# GridSearchCV를 이용하여 하이퍼패러미터 튜닝
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=kfold,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

# 학습 
print(grid_search.fit(X,y))

import pandas as pd
df=pd.DataFrame(grid_search.cv_results_)
print(df)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(df['param_knn__n_neighbors'], -df['mean_train_score'])
plt.plot(df['param_knn__n_neighbors'], -df['mean_test_score'])
plt.show()

print(grid_search.best_params_, grid_search.best_score_)

# 데이터 전처리
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/heart.csv')
print(df)

df = df.dropna()
df = pd.get_dummies(df, columns=['Sex', 'AHD', 'ChestPain', 'Thal', 'AHD'])
print(df)

y = df['AHD_Yes']
X = df.drop(columns=['AHD_No', 'AHD_Yes'])

# 데이터 스케일링
X_norm = (X-X.mean())/X.std()
print(X_norm)

from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
kfold = KFold(n_splits=5, shuffle=True)

train_accuracy = []
test_accuracy = []
for i in range(2, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_norm, y)
    result = cross_validate(X=X_norm, y=y, cv=kfold,
                            scoring=['accuracy', 'roc_auc'],
                            estimator=knn,
                            return_train_score=True)
    print(i, result['test_accuracy'].mean(), result['test_roc_auc'].mean())
    train_accuracy.append(result['train_accuracy'].mean())
    test_accuracy.append(result['test_accuracy'].mean())

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(train_accuracy, label='train')
plt.plot(test_accuracy, label='test')
plt.legend()
plt.show()   

# GridSearchCV를 사용한 자동화
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV

# 파이프라인 재료
knn = KNeighborsClassifier()
scaler = StandardScaler()

# 파이프라인 만들기
pipe = Pipeline([('scaler',scaler),('knn',knn)])

# 하이퍼패러미터 범주지정
degree = range(1,20)
param_grid = {'knn__n_neighbors':degree}

# 교차검증 객체지정
kfold = KFold(n_splits=5, shuffle=True)

# GridSearchCV이용 하이퍼패러미터 튜닝 자동화
grid_search = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid,
    cv = kfold,
    scoring = 'accuracy',
    return_train_score=True    
)

# 학습
print(grid_search.fit(X,y))

import pandas as pd

df=pd.DataFrame(grid_search.cv_results_)
print(df)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(df['param_knn__n_neighbors'],df['mean_train_score'])
plt.plot(df['param_knn__n_neighbors'],df['mean_test_score'])
plt.show()

print(grid_search.best_params_, grid_search.best_score_)