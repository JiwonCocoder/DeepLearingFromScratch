import numpy as np
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        #순전파시, 편향은 각각의 데이터(1번째 데이터,2번째 데이터..)에 더해짐
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        #역전파시, 각 데이터의 역전파의 값이 편향의 원소에 모여야 함.
        #그래서 0번째 축, 데이터를 단위로 한 축,axis = 0의 총합을 구하는 것.
        self.db = np.sum(dout, axis = 0)

        return dx

