# -*- coding: utf-8 -*-\
'''
이미 학습이 끝난 신경망을 이용하여
mnist data에 대한 분류(추론)을
mini_batch로 실시

'''
import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from mnistClassification import init_network, predict

#정답 레이블이 2나 7등의 숫자 레이블로 주어졌을 때
def numeric_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
'''
np.arange(batch_size): 0부터 batch_size -1 까지 배열을 생성.ex[0, 1, 2,.....batch_size -1 ]
t: 레이블이 [2, 7, 0, 9, 4]와 같이 저장되어 있으므로
y[np.arange(batch_size), t] :각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출. ex. y[0,2], y[1,7] 
'''

#정답 레이블이 원-핫 인코딩: t가 0인 원소의 교차 엔트로피도 0이라서, 그 계산은 무시 가능.
#그래서 t*np.log(y)해도 괜찮았다.
def cross_entropy_error(y, t):
    #y가 1차원일땐, []이 하나만 존재하는 상황 데이터 하나당 교차 엔트로피 오차를 구해야할 때 reshape를 이용해서 형상을 바꿔준다.
    if y.ndim ==1:
        #t.size나 y.size는 클래스수를 의미
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    #데이터가 배치로 묶여 입력될 경우
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size



(x_train, t_train), (x_test, t_test) = \
load_mnist(normalize = True, one_hot_label = True)


print(x_train.shape)
print(t_train.shape)
print(x_train.size)
print(60000 * 784)
train_size = x_train.shape[0]
batch_size = 1
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch)
print(t_batch)
network = init_network()
y_batch = predict(network, x_batch)
print(y_batch.shape)
loss_function=cross_entropy_error(y_batch, t_batch)
print(loss_function)
'''minibatch
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
'''