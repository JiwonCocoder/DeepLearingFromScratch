import numpy as np

def softmax_function(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#정답 레이블이 원-핫 인코딩: t가 0인 원소의 교차 엔트로피도 0이라서, 그 계산은 무시 가능.
#그래서 t*np.log(y)해도 괜찮았다.
def cross_entropy_error(y, t):
    #y가 1차원 = 데이터 하나당 교차 엔트로피 오차를 구하는 경우
    if y.ndim ==1:
        #총 샙플수(1), t.size = 1 X 클래스 수
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    #데이터가 배치로 묶여 입력될 경우
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
#다차원 배열도 사용 가능하게 만든 버전
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #x와 형상이 같은 배열을 생성.
    #추가된 부분
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
   # for idx in range(x.size):
        idx = it.multi_index
        temp_val = x[idx]
        #f(x+h)계산
        x[idx] = temp_val + h
        fxh1 = f(x)
        #f(x-h)계산
        x[idx] = temp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] =temp_val #x값 복원
        it.iternext()
    return grad

'''
간단한 신경망을 통해, loss function(교차 엔트로피 오차함수)를 구현.
'''
def f(W):
    return net.loss(x,t)
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #2X3형상을 가진 가중치 매개변수를 정규분포로 초기화
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax_function(z)
        loss = cross_entropy_error(y, t)
        return loss
net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
t = np.array([0, 0, 1])
dW = numerical_gradient(f, net.W)
print(dW)