import sys, os
import numpy as np
sys.path.append(os.pardir)
from functions import *
def sigmoid(x):
    y = 1 / (1 +np.exp(-x))
    return y
def softmax(x):
    c = np.max(x)
    up = np.exp(x- c)
    down = np.sum(up)
    y = up/down
    return y
def cross_entropy_error(y, t):
    if y.ndim ==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    #훈련데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y*[np.arange(batch_size), t] + 1e-7)) /batch_size
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

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        #클래스의 인스턴스 변수1:params-신경망의 매개변수를 보관하는 딕셔너리 변수
        self.params = {} #빈 dictionary 만들고
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #(input갯수 x 은닉층 input 갯수)배열을 만든 후, wieght_init_std(초기 표준편차)를 곱해준다.
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)#(은닉층 input 갯수 x 출력층 input 갯수)배열을 만든 후, wieght_init_std(초기 표준편차)를 곱해준다.
        self.params['b2'] = np.zeros(output_size)
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    #x: 입력 데이터, t:정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis =1)
        accuracy = np.sum(y == t) /float(x.shape[0])
        return accuracy
    #x:입력데이터, t:정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        #클래스의 인스턴스 변수2: grads -numberical_gradient()메서드의 반환값으로 기울기를 보관하는 딕셔너리 변수
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads