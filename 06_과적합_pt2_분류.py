# 데이터 그리기
# 주피터 노트북에서 경고메시지 안나오게 설정
import warnings
warnings.filterwarnings('ignore')

# 데이터 불러오기
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = None
import numpy as np
np.random.seed(42)
df = pd.read_csv("https://raw.githubusercontent.com/tonyfischetti/InteractiveLogisticRegression/master/moons.csv")
print(df)

# 레이블에 따라 서로 다른 색깔로 표현
import seaborn as sns # matplotlib의 업그레이드
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6)) # 사이즈 지정
sns.scatterplot(x = 'X1', y = 'X2', hue='Y', data=df) # hue : 데이터의 레이블(y값)
plt.show()

# 독립 종속변수 나누기
X = df[['X1', 'X2']]
y = df['Y']

# 데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
# 7:3 분할(훈련 : 테스트, 무작위)

plt.figure(figsize=(8, 6))
sns.scatterplot(x = X_train['X1'], y = X_train['X2'], hue=y_train)
plt.show()

# 로지스틱회귀
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

def plot_Classifier(degree, graph=True):

    # 모델 생성
    clf = LogisticRegression()

    # 데이터 변환
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_ = poly.fit_transform(X_train)

    # 모델학습
    clf.fit(X_train_, y_train)
    y_pred_train = clf.predict(X_train_)

    # 그래프 그리기(몰라도 됨)
    if graph:
        X1_grid, X2_grid = np.meshgrid(np.arange(-2, 3, 0.01), np.arange(-1.5, 2, 0.01))
        # -2~3까지 0.01단위로 격자로 쪼갬, -1.5~2까지 0.01단위로 격자로 쪼갬
        X_bound = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))
        # colum_stack을 합침
        X_bound_ = poly.fit_transform(X_bound)
        y_bound = clf.predict(X_bound_)
        # 위에서 만든 격자 데이터을 대입, 학습한 모델로 분류(0, 1)
        y_bound = y_bound.reshape(X1_grid.shape)

        # create a contour plot with colored contours based in the density of the 'X1' and 'X2' columns
        sns.scatterplot(x = X_test['X1'], y = X_test['X2'], hue=y_test)
        plt.contourf(X1_grid, X2_grid, y_bound, cmap='RdBu', alpha=0.3) # 의사결정 경계선 그리는 함수
        
    # Show the plot
    plt.show()

    # 분류지표 계산
    X_test_ = poly.fit_transform(X_test)
    y_pred_test = clf.predict(X_test_)
    train_acc = accuracy_score(y_train, y_pred_train) # 훈련 데이터에 대한 분류 정확도
    test_acc = accuracy_score(y_test, y_pred_test) # 테스팅 데이터에 대한 분류 정확도
    print(f'training ACC : {train_acc}')
    print(f'testing ACC: {test_acc}')

    return train_acc, test_acc

# 1차 로지스틱회귀
plot_Classifier(1)

# 2차 로지스틱회귀
plot_Classifier(2)

# 5차 로지스틱회귀
plot_Classifier(5)

# 8차 로지스틱회귀
plot_Classifier(8)

# 15차 로지스틱회귀
plot_Classifier(15)

# 유연성 높은 모델이 반드시 좋은가?
train_acc=[]
test_acc=[]
degrees=range(1,16)

for i in degrees:
    print("="*10 + str(i) + "="*10)
    train, test = plot_Classifier(i, graph=False)
    train_acc.append(train)
    test_acc.append(test)

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_acc, label = 'train_accuracy')
plt.plot(degrees, test_acc, label='test_accuracy')
plt.legend()
plt.show()    