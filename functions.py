import numpy as np
import matplotlib.pylab as plt

#각층에 필요한 가중치와 편향을 초기화. 이들을 딕셔너리 변수인 network에 저장.
def init_network():
    network = {} #딕셔너리 변수를 만들고
    #딕셔너리 쌍을 추가시킴
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network
# -*- coding: utf-8 -*-

#입력신호를 출력으로 전달됨(순전파)
def forward(network, x):
    #J)W들과 b들은 넘파이배열형의 변수일 것.
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y
def softmax_function(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def identity_function(x):
    return x

def ReLU_function(x):
    return np.maximum(0, x)

def sigmoid_function(x):
    y = 1 / (1 + np.exp(-x))
    return y

def step_function(x):
    y = x>0
    return y.astype(np.int)


'''
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
#x = np.arange(-5.0, 5.0, 0.1)
print(x.shape)

#y = ReLU_function(x)
#y = sigmoid_function(x)
#y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()
'''