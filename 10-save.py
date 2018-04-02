"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 10 - save

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model   # 读取模型的模块

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)

# save
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

# load
model = load_model('my_model.h5')
print('test after load: ', model.predict(X_test[0:2]))
"""
# 只保存weights
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

# 只保存神经网络的结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
"""



