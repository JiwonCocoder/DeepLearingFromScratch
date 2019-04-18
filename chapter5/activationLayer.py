import numpy as np
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

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        return out
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
