# %%
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from dezero import Variable
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.sin(x)
# print(y)
y.backward(create_graph=True)
# print(x.grad)
# print(y)

for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)
    

# %%
import numpy as np
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ['y=sin(x)', "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])

plt.legend(loc='lower right')
plt.show()







# %%
# import numpy as np
# import matplotlib.pyplot as plt

# # 定义 sin 函数
# def f(x):
#     return np.sin(x)

# # 计算一阶导数
# def f_prime(x):
#     return np.cos(x)

# # 计算二阶导数
# def f_double_prime(x):
#     return -np.sin(x)

# # 计算三阶导数
# def f_triple_prime(x):
#     return -np.cos(x)

# # 生成 x 值
# x = np.linspace(0, 2 * np.pi, 1000)

# # 计算对应的 y 值
# y = f(x)
# y_prime = f_prime(x)
# y_double_prime = f_double_prime(x)
# y_triple_prime = f_triple_prime(x)

# # 绘制图像
# plt.figure(figsize=(6, 10))  # 设置图像大小

# plt.subplot(4, 1, 1)
# plt.plot(x, y)
# plt.title('sin(x)')

# plt.subplot(4, 1, 2)
# plt.plot(x, y_prime)
# plt.title('cos(x) - First Derivative')

# plt.subplot(4, 1, 3)
# plt.plot(x, y_double_prime)
# plt.title('-sin(x) - Second Derivative')

# plt.subplot(4, 1, 4)
# plt.plot(x, y_triple_prime)
# plt.title('-cos(x) - Third Derivative')

# plt.tight_layout()
# plt.show()
# # %%
