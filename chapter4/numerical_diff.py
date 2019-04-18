import numpy as np
def function_2(x):
     return x[0]**2 + x[1]**2
def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h)) / (2*h)
#f:함수, x는 넘파이 배열 따라서 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구함.
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #x와 형상이 같은 배열을 생성.

    for idx in range(x.size):
        temp_val = x[idx]
        #f(x+h)계산
        x[idx] = temp_val + h
        fxh1 = f(x)
        #f(x-h)계산
        x[idx] = temp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] =temp_val #x값 복원
    return grad
#f:최적화하려는 함수, init_x:초기값, step_num:경사법으로 매개변수 갱신을 몇번할지
def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    return x
print (numerical_gradient(function_2, np.array([3.0, 4.0])))
print(gradient_descent(function_2, np.array([-3.0, 4.0]), lr = 1e-10, step_num = 100))
