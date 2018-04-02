import numpy as np
np.random.seed(123)
from keras.models import Sequential  # 顺序建立的神经网络
from keras.layers import Dense    # 全连接层
import matplotlib.pyplot as plt

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
y = 0.5*X + 2 + np.random.normal(0, 0.05, (200,))


plt.scatter(X, y)
plt.show()


X_train, y_train = X[:160], y[:160]
X_test, y_test = X[160:], y[160:]


model = Sequential()
model.add(Dense(units=1, input_dim=1))   # 本层的输出神元数units=1, 输入维度为1
# model.add(Dense(units=10,))   # 如果有第二层,默认上一层的输出,就是本层的输入

model.compile(loss='mse', optimizer='sgd')   # mse,二次方的损失函数;sgd,随机梯度下降


print('开始训练...')
for step in range(300):
    cost = model.train_on_batch(X_train, y_train)
    if step%100 ==0:
        print('train cost:', cost)


print('测试...')
cost = model.evaluate(X_test, y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

y_pred = model.predict(X_test)
plt.scatter(X_test, y_test)  # 真实点
plt.plot(X_test, y_pred)    # 预测线
plt.show()










