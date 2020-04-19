'''
@Author: wallfacer (Yanhan Zhang)
@Time: 2020/4/19 3:08 PM
'''

import keras
from keras import backend as K
import numpy as np
from data_processor import data_processor


def root_mean_squared_error(y_true, y_pred):
    print(y_pred)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class rater:
    def __init__(self, train_x, train_y, test_x, test_y, matrix, batch_size=32, epochs=10):
        self.train_x = train_x
        self.train_y = np.array(train_y, dtype=np.float)
        self.test_x = test_x
        self.test_y = np.array(test_y, dtype=np.float)
        self.matrix = matrix

        self.train_x = np.array(list(map(lambda l: l[:23] + l[24:], self.train_x)))
        self.test_x = np.array(list(map(lambda l: l[:23] + l[24:], self.test_x)))
        self.normalization()

        self.model = keras.models.Sequential()
        self.batch_size = batch_size
        self.epochs = epochs

    def normalization(self):

        self.test_x = (self.test_x - np.min(self.train_x, axis=0)) / (
                np.max(self.train_x, axis=0) - np.min(self.train_x, axis=0))
        self.train_x = (self.train_x - np.min(self.train_x, axis=0)) / (
                    np.max(self.train_x, axis=0) - np.min(self.train_x, axis=0))

    def standardization(self):
        mean = np.mean(self.train_x, axis=0)
        sigma = np.std(self.train_x, axis=0)
        self.train_x = (self.train_x - mean) / sigma
        self.test_x = (self.test_x - mean) / sigma

    def build_model(self):
        input_shape = (45,)
        self.model.add(keras.layers.Dense(16, activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(1))
        self.model.compile(loss='mse', optimizer='adamax', metrics=['accuracy'])

    def train(self):
        self.history = self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epochs)

    def test(self):
        self.score = self.model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size)
        print(self.score)

    def check(self):
        for i in range(self.train_x.shape[0]):
            for j in range(self.train_x.shape[1]):
                if self.train_x[i][j] < 0 or self.train_x[i][j] > 1:
                    print(i, j, self.train_x[i][j])


if __name__ == '__main__':
    dp = data_processor(train_file='ml-100k/ua.base', test_file='ml-100k/ua.test')
    r = rater(dp.get_train_data(), dp.get_train_label(), dp.get_test_data(), dp.get_test_label(), dp.get_matrix(),
              batch_size=100000, epochs=500)
    r.check()
    r.build_model()
    r.train()
    r.test()
    print('OK')
