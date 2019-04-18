import sys, os
import numpy as np
sys.path.append(os.pardir)
from collections import OrderedDict
#softmax 뒤에 손실함수인 교차 엔트로피 오차를 포함하여 하나의 layer로 만든것

'''
+backpropa:softmax-with-loss 계층
'''


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

def sigmoid(x):
    y = 1 / (1 +np.exp(-x))
    return y
def softmax(x):
    c = np.max(x)
    up = np.exp(x- c)
    down = np.sum(up)
    y = up/down
    return y
class Relu:
    def __init__(self):
        #역전파시, mask값을 이용해 전파여부를 결정하게 되니까 변수로 지정해둠
        self.mask = None

    def forward(self,x):
        '''
        x <= 0일때, mask값 true되도록
        순전파시 true이면 스위치 OFF되듯 하류에게 신호전파 X
        일단 x값을 copy한 뒤, mask값이 true인 index의 x값에 해당하는 out = 0으로 assign
        '''
        self.mask = (x <=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        #mask가 True인 dout은 0으로 assign으로 바꿔줌. 하류로 전파되지 않는다.
        dout[self.mask] = 0
        dx = dout
        return dx
    #현재 정답데이터가 숫자로 표기되어 있는 상
    #[batch_index, 정답y 인덱스]에 있는 확률에 * 정답y를 곱한 값을 log취한다
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
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


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        #클래스의 인스턴스 변수1:params-신경망의 매개변수를 보관하는 딕셔너리 변수
        self.params = {} #빈 dictionary 만들고
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #(input갯수 x 은닉층 input 갯수)배열을 만든 후, wieght_init_std(초기 표준편차)를 곱해준다.
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)#(은닉층 input 갯수 x 출력층 input 갯수)배열을 만든 후, wieght_init_std(초기 표준편차)를 곱해준다.
        self.params['b2'] = np.zeros(output_size)

        '''
        +backpropa:계층생성
        '''
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
        '''
        #np.dot을 수행하는 AffineLayer를 새로 만들었기때문에
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    #x: 입력 데이터, t:정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
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

    def gradient(self, x, t):
        #순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads