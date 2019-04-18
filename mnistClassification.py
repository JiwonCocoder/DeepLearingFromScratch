# -*- coding: utf-8 -*-

'''
신경망의 추론처리.

'''
import sys, os
sys.path.append(os.pardir)
#부모디렉터리에 있는 mnist.py의 load_mnist함수를 임포터.
import numpy as np
from mnist import load_mnist
from PIL import Image
import pickle
from functions import sigmoid_function, softmax_function
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
load_mnist(flatten=True, normalize = False)
    print("y_shape"+ str(t_test.shape))
    print("x_shape"+ str(x_test.shape))
    return x_test, t_test
#이미 학습된 가중치 매개변수를 읽어와서 network를 구성함.
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

        return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_function(a3)

    return y

x, t =get_data()
network = init_network()

batch_size =100
accuracy_cnt = 0
'''
axis=1: 100X10의 배열 중 1번째 차원을 구성하는 각 원소에서(1번째 차원을 축으로) 
최댓값 인덱스를 찾도록 한 것. 
즉 각각의 1X10에서 max찾는 것. 
인덱스가 0부터 시작하니 0번째 차원이 가장 처음 차원.
'''
test = np.array([[0.1, 0.8, 0.1], #1
                 [0.3, 0.1, 0.6], #2
                 [0.2, 0.5,0.3], #1
                 [0.8,0.1,0.1]]) #0
print(test.ndim)
ytest= np.argmax(test, axis =1)

print(ytest)
#배치 단위로 분류한 결과를 실제 답과 비교
y1 = np.array([1, 2, 0, 0])
t1 = np.array([1, 2, 1, 0])
print(y1==t1)

'''minibatch
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
'''


for i in range(len(x)):
    y = predict(network, x[i])
    print(y.shape)
    print(y.shape[0])
    print(y.ndim)
    #np 배열인 y중에서 max인 값을 p로 설정.
    p = np.argmax(y)
    if p ==t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))



'''1차원 넘파이 배열데이터를 이미지로 표시
img = img.reshape(28, 28)
img_show(img)
'''

