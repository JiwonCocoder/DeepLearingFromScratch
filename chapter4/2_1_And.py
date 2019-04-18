# -*- coding: utf-8 -*-

import numpy as np
x = np.array([0,1])
w = np.array([0.5, 0.5])
b = -0.7

y = w*x + b
print(y)
#np메서드는 입력한 배열에 담긴 모든 원소의 총합을 계산
y = np.sum(w*x) + b
print(y)
print(0.5 - 0.7)