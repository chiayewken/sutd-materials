# Theory Homework Week 8
Chia Yew Ken (1002675)

## Task 1
### Is c(t−1) a function of h(t−1)?
```
No, because c depends on h values from previous time steps, but not 
the current time step:

c(t) depends on c(t-1), u(t),
c(t-1) depends on c(t-2), u(t-1),
u(t) depends on x(t), h(t-1),
h(t-1) depends on o(t-1), c(t-1),
o(t-1) depends on h(t-2).
```
### Write down the derivative dh(t)/dh(t-1)
```
dh(t)/dh(t-1) = a+b+c+d
where
a = tanh(c(t)).o`
b = o(t).sech**2(c(t)).c(t-1).f`
c = o(t).sech**2(c(t)).u(t).i`
d = o(t).sech**2(c(t)).i(t).u`
where
o`,f`,i`,u` refer to derivative with respect to h(t-1)
```
### Calculate the derivative of the sigmoid
```Derivative of sigmoid(x) = sigmoid(x).(1-sigmoid(x))```

### Write down the derivative df(t)/dh(t−1)
```x.(1-x).Uf where x=sigmoid(Wf.x(t) + Uf.h(t-1))```

### Find h(t-1) that maximizes f(t)(d) given h.h = 1
```
Wf(d).x(t) is zero so that does not affect arg-max
Sigmoid also doesn't affect arg-max
Only Uf(d).h(t-1) affects arg-max
To maximize dot product, Uf(d) and h(t-1) should be parallel
Given ||h(t-1)|| = 1, h(t-1) should be = Uf(d) / ||Uf(d)||
```
### Does the arg-max or max depend on the value of Wf(d).x(t)?
```
No, because x(t) = 0 so Wf(d).x(t) = 0
```

## Task 2
### What is the spatial size of output feature map from the convolution?
```
O = floor((I - K + 2P) / S) + 1
Output height = floor((78 - 5 + 2*2) / 3) + 1 = 26
Output width = floor((84 - 5 + 2*2) / 3) + 1 = 28
Thus output size = (26, 28)
```
### What spatial input size do you need so spatial output size is 16?
```
O = floor((I - K + 2P) / S) + 1
O - 1 = floor((I - K + 2P) / S)
(O - 1) * S = I - K + 2P
I = (O - 1)*S + K - 2P = 15*3 + 9 - 2 = 52
```
### How many trainable parameters are:
* In a 2-D convolutional layer with input (32, 19, 19), kernel size (7, 7), stride
3, 64 kernel channels, no padding, no bias term?
    * ```Num parameters = K*K*C*N = 7*7*32*64 = 100352```
* How many multiplications and how many additions are performed in this
case above?
    * ```Output size = floor((I-K+2P)/S)+1 = 5```
    * ```Num multiply = Num params * 5 * 5 = 100352 * 5 * 5 = 2508800```
    * ```Num add = K*K*5*5*(N-1)*C = 7*7*5*5*31*64 = 2507200```
* In a 2-D convolutional layer with input (512, 25, 25), kernel size (1, 1),
stride 1, 128 kernel channels, padding 2, no bias term?
    * ```Num parameters = K*K*C*N = 1*1*128*512 = 65536```

    
    


