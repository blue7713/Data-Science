import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# 데이터 만들기
X = np.linspace(-5, 3, 100) # -5~3까지 100개의 등간격 데이터
y = 0.1*(X-3)*(X+3)*(X+1)*(X+5)+ np.random.normal(1, size=100)
plt.scatter(X, y)
plt.show()

# Cross Validate 사용
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression

reg=LinearRegression() # 학습할 모형
kfold = KFold(n_splits=5, shuffle=True) # 데이터 쪼갤 방식

# https://scikit-learn.org/stable/modules/model_evaluation.html
metrics = ['neg_mean_absolute_error','neg_mean_squared_error','r2'] # 교차검증하면서 기록할 평가지표

result = cross_validate(X=X.reshape(-1, 1), # X데이터
                        y=y, # y데이터
                        estimator=reg, # 학습시킬 모델
                        scoring=metrics, # 평가할 지표
                        cv=kfold, # 교차검증 방식
                        return_train_score=True)

print(result)

# 결과의 사용
test_mae = -result['test_neg_mean_absolute_error'].mean()
test_mse = -result['test_neg_mean_squared_error'].mean()
test_r2 = result['test_r2'].mean()
print(test_mae, test_mse, test_r2)