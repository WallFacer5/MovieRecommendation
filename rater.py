'''
@Author: wallfacer (Yanhan Zhang)
@Time: 2020/4/19 3:08 PM
'''

import keras
from keras import backend as K
import numpy as np
import pickle
from data_processor import data_processor1m


def root_mean_squared_error(y_true, y_pred):
    print(y_pred)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class rater:
    def __init__(self, train_x, train_y, test_x, test_y, matrix, batch_size=32, epochs=10):
        self.matrix = matrix

        self.train_sex = np.array(list(map(lambda d: d['sex'], train_x))).reshape([-1, 1])
        self.test_sex = np.array(list(map(lambda d: d['sex'], test_x))).reshape([-1, 1])

        self.train_age = np.array(list(map(lambda d: d['age'], train_x))).reshape([-1, 1])
        self.test_age = np.array(list(map(lambda d: d['age'], test_x))).reshape([-1, 1])

        self.train_occupation = np.array(list(map(lambda d: d['occupation'], train_x))).reshape([-1, 1])
        self.test_occupation = np.array(list(map(lambda d: d['occupation'], test_x))).reshape([-1, 1])

        self.train_zip = np.array(list(map(lambda d: d['zip'], train_x))).reshape([-1, 1])
        self.test_zip = np.array(list(map(lambda d: d['zip'], test_x))).reshape([-1, 1])

        self.train_title = np.array(list(map(lambda d: d['title'], train_x)))
        self.test_title = np.array(list(map(lambda d: d['title'], test_x)))

        self.train_year = np.array(list(map(lambda d: d['year'], train_x))).reshape([-1, 1])
        self.test_year = np.array(list(map(lambda d: d['year'], test_x))).reshape([-1, 1])

        self.train_genre = np.array(list(map(lambda d: np.array(d['genre']), train_x)))
        self.test_genre = np.array(list(map(lambda d: np.array(d['genre']), test_x)))

        self.train_age, self.test_age = self.normalization(self.train_age, self.test_age)
        self.train_year, self.test_year = self.normalization(self.train_year, self.test_year)

        self.train_y = train_y
        self.test_y = test_y

        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs

    def normalization(self, train_v, test_v):
        test_v = (test_v - np.min(train_v, axis=0)) / (
                np.max(train_v, axis=0) - np.min(train_v, axis=0))
        train_v = (train_v - np.min(train_v, axis=0)) / (
                np.max(train_v, axis=0) - np.min(train_v, axis=0))
        return train_v, test_v

    def standardization(self):
        mean = np.mean(self.train_x, axis=0)
        sigma = np.std(self.train_x, axis=0)
        self.train_x = (self.train_x - mean) / sigma
        self.test_x = (self.test_x - mean) / sigma

    def build_model(self):
        input_sex = keras.layers.Input(shape=(1,))
        input_age = keras.layers.Input(shape=(1,))
        input_occupation = keras.layers.Input(shape=(1,))
        input_zip = keras.layers.Input(shape=(1,))
        input_title = keras.layers.Input(shape=(768,))
        input_year = keras.layers.Input(shape=(1,))
        input_genre = keras.layers.Input(shape=(18,))

        occupation_emb = keras.layers.Embedding(21, 4, input_length=1)(input_occupation)
        reshape1 = keras.layers.Reshape((-1,))(occupation_emb)
        zip_emb = keras.layers.Embedding(3440, 32, input_length=1)(input_zip)
        reshape2 = keras.layers.Reshape((-1,))(zip_emb)

        concat = keras.layers.Concatenate()(
            [input_sex, input_age, reshape1, reshape2, input_title, input_year, input_genre])

        dense1 = keras.layers.Dense(256)(concat)
        act1 = keras.layers.Activation('relu')(dense1)
        dense2 = keras.layers.Dense(16)(act1)
        act2 = keras.layers.Activation('relu')(dense2)
        pred = keras.layers.Dense(1)(act2)

        self.model = keras.models.Model(
            inputs=[input_sex, input_age, input_occupation, input_zip, input_title, input_year, input_genre],
            outputs=pred)

        self.model.compile(loss=root_mean_squared_error, optimizer='adamax', metrics=['accuracy'])

    def train(self):
        self.history = self.model.fit(
            [self.train_sex, self.train_age, self.train_occupation, self.train_zip, self.train_title, self.train_year,
             self.train_genre],
            self.train_y, batch_size=self.batch_size, epochs=self.epochs)

    def test(self):
        self.score = self.model.evaluate(
            [self.test_sex, self.test_age, self.test_occupation, self.test_zip, self.test_title, self.test_year,
             self.test_genre],
            self.test_y, batch_size=self.batch_size)
        print(self.score)


if __name__ == '__main__':
    # dp = data_processor1m()
    data = pickle.load(open('data1m.pkl', 'rb'))
    r = rater(data[0], data[2], data[1], data[3], data[4], batch_size=256, epochs=5)
    r.build_model()
    r.train()
    r.test()
    print('OK')
