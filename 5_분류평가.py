# 분류모형의 평가
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
df = pd.read_csv("https://raw.githubusercontent.com/ralbu85/DataScience_2022S/master/data/classification_data.csv")
print(df)

# 시각화
import matplotlib.pyplot as plt
plt.scatter(df['X1'], df['X2'], c=df['y'])
plt.show()

# 로지스틱회귀 학습
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
X = df[['X1', 'X2']]
y = df['y']

# 학습
reg.fit(X, y)

# 평가지표
# Accuracy
# Precision
# Recall
# ROC

# 정확도
# 모델이 예측한 값(Numpy Array)
y_pred = reg.predict(X)

# 모델이 맞추어야 하는 정답(Pandas Series)
y

import numpy as np
accuracy = np.sum(y_pred == y)/y.size # numpy sum은 true를 1로 false를 0으로 본다
print(accuracy)

# 정밀도
# 모델이 참이라고 예측한 데이터 중에 실제로 참인것의 비율
df['y_pred'] = y_pred # y_pred 열 추가
print(df)

# True Positive
tp = len(df[(df['y_pred'] == 1) & (df['y'] == 1)])
print(tp)

# True Negative
tn = len(df[(df['y_pred'] == 0) & (df['y'] == 0)])
print(tn)

# False Positive
fp = len(df[(df['y_pred'] == 1) & (df['y'] == 0)])
print(fp)

# False Negative
fn = len(df[(df['y_pred'] == 0) & (df['y'] == 1)])
print(fn)

# Accuracy
accuracy = (tp + tn)/(tp + tn + fp + fn)
print(accuracy)

# Precision
precision = tp/(tp + fp)
print(precision)

# Recall
recall = tp/(tp + fn)
print(recall)

# sklearn을 이용한 confusion matrix 구하기
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))

# ROC 구해보기
# 의사결정의 기준확률을 0부터 1까지 변경시켜가면서 
# true_positive_rate와 false_positive_rate의 위치를
# 그래프로 그린 것
# ROC의 너비 ~ 1: 완벽한 불류모형
# ROC의 너비 ~ 0.5: 잘못된 분류모형


# 의사결정 기준 th에 따라 y_pred를 구하는 함수
def predict_th(reg, th):
    result = []
    for i in reg.predict_proba(X)[:,1]:
        if i > th:
            result.append(1)
        else:
            result.append(0)    
    return np.array(result) 

# 의사결정 기준 th에 따라 tpr과 fpr을 얻는 함수
def get_tpr_fpr(reg, th):
    y_hat = predict_th(reg, 0.7) # 의사결정 기준치 0.7
    cm = confusion_matrix(y, y_hat)

    tp, fp, tn, fn = cm[1][1], cm[0][1], cm[0][0], cm[1][0]

    tp_rate = tp / (tp + fn)
    fp_rate = fp / (tn + fp)
    print(tp_rate)
    print(fp_rate)
    return tp_rate, fp_rate

get_tpr_fpr(reg, 0.7)

tpr_result = []
fpr_result = []
for th in np.linspace(0, 1, 100): # 0~1사이의 값을 등간격으로 100개 대입
    tpr, fpr = get_tpr_fpr(reg, th)
    tpr_result.append(tpr)
    fpr_result.append(fpr)
  

# ROC
plt.plot(fpr_result, tpr_result, color="red")
plt.show()

# sklearn을 이용하여 ROC의 면적 계산하기
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
print('roc: ', roc_auc_score(y, reg.predict_proba(X)[:,1]))
print('precision: ', precision_score(y, y_pred))
print('recall: ', recall_score(y, y_pred))
print('accuracy: ', accuracy_score(y, y_pred))