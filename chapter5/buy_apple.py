#곱셈 계층
class MulLayer:
    #초기화:순전파시 입력값을 유지하기 위해, 입력변수를 생성
    def __init__(self):
        self.x = None
        self.y = None
#순전파:수전파의 입력 값 x,y를 assign해주고, 곱한 후 그 값을 반환
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
#상류에서 넘어온 미분(dout)에 순전파 때 assign한 값을 '서로 바꿔'곱한 후 하류로
    def backward(self, dout):
        #x와 y를 바꾼다.
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    # 덧셈 계층은 순전파 입력값을 저장할 필요없으니까
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    #역전파시, 상류값을 하류에 전달해주기만 하면 되니까
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
mul_tax_layer = MulLayer()

add_apple_orange_layer = AddLayer()

#순전파: 각 곱셈노드에 대한 입력신호(x,y)를 저장.
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
fruit_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(fruit_price, tax)
print(price)

#역전파:순전파 순서와 반대
#backward가 받는 인수는 '순전파의 출력에 대한 미분'을 받아야
dprice = 1
dfruit_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dfruit_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(dapple, dapple_num, dorange, dorange_num, dtax)