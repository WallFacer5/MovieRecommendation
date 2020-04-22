'''
@Author: wallfacer (Yanhan Zhang)
@Time: 2020/4/18 6:44 PM
'''

import numpy as np
from bert_serving.client import BertClient
from sklearn.model_selection import train_test_split
import pickle


class data_processor:
    def __init__(self, occupations_file='ml-100k/u.occupation', users_file='ml-100k/u.user',
                 genres_file='ml-100k/u.genre', items_file='ml-100k/u.item', train_file='ml-100k/u1.base',
                 test_file='ml-100k/u1.test'):
        self.occupations = []
        self.occupations_dict = {}
        self.genres = []
        self.genres_dict = {}
        self.users = [[]]
        self.items = [[]]
        self.sex_dict = {'M': 1, 'F': 0}
        self.zipcode_dict = {}
        self.month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        self.load_users(occupations_file, users_file)
        self.load_items(genres_file, items_file)
        self.matrix = [[0 for j in range(len(self.items))] for i in range(len(self.users))]
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.load_train_data(train_file)
        self.load_test_data(test_file)
        self.train_formatted = []
        self.test_formatted = []
        self.concat_format()

    def gen_onehot(self, i, length):
        onehot = [0 for j in range(length)]
        onehot[i] = 1
        return onehot

    def load_occupations(self, occupations_file):
        occupations_file = open(occupations_file, 'r')
        i = 0
        for line in occupations_file:
            line = line.strip()
            if line != '':
                self.occupations.append(line)
                self.occupations_dict[line] = i
                i += 1
        occupations_file.close()

    def load_users(self, occupations_file, users_file):
        self.load_occupations(occupations_file)
        users_file = open(users_file, 'r')
        for line in users_file:
            line = line.strip().split('|')
            if line != []:
                line[1] = int(line[1])
                line[2] = self.sex_dict[line[2]]
                line[3] = self.occupations_dict[line[3]]
                try:
                    line[4] = self.zipcode_dict[line[4]]
                except Exception:
                    self.zipcode_dict[line[4]] = len(self.zipcode_dict)
                    line[4] = len(self.zipcode_dict)
                self.users.append(line[1:])
        users_file.close()
        for i, user in enumerate(self.users):
            try:
                user[2:] = self.gen_onehot(user[2], len(self.occupations))
                self.users[i] = user
            except Exception:
                pass

    def load_genres(self, genres_file):
        genres_file = open(genres_file, 'r')
        for line in genres_file:
            line = line.strip()
            if line != '':
                line = line.split('|')
                self.genres.append(line[0])
                self.genres_dict[line[0]] = int(line[1])
        genres_file.close()

    def load_items(self, genres_file, items_file):
        self.load_genres(genres_file)
        items_file = open(items_file, 'r', encoding='latin')
        for line in items_file:
            line = line.strip().split('|')
            if line != []:
                line.pop(3)
                line.pop(3)
                if ' ' in line[1]:
                    line[1] = ' '.join(line[1].split(' ')[:-1])
                try:
                    line[2] = line[2].split('-')
                    line[2][1] = self.month_map[line[2][1]]
                except Exception:
                    line[2] = ['1', 1, '1997']
                line[2:3] = line[2]
                for j in range(len(line)):
                    try:
                        line[j] = int(line[j])
                    except Exception:
                        pass
                self.items.append(line[1:])
        items_file.close()

    def load_train_data(self, train_file):
        train_file = open(train_file, 'r')
        for line in train_file:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                self.train_data.append([int(line[0]), int(line[1])])
                self.train_label.append(int(line[2]))
                self.matrix[int(line[0])][int(line[1])] = int(line[2])
        train_file.close()

    def load_test_data(self, test_file):
        test_file = open(test_file, 'r')
        for line in test_file:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                self.test_data.append([int(line[0]), int(line[1])])
                self.test_label.append(int(line[2]))
        test_file.close()

    def concat_format(self):
        for i in range(len(self.train_data)):
            self.train_formatted.append(self.users[self.train_data[i][0]] + self.items[self.train_data[i][1]])
        for i in range(len(self.test_data)):
            self.test_formatted.append(self.users[self.test_data[i][0]] + self.items[self.test_data[i][1]])

    def get_matrix(self):
        return self.matrix

    def get_train_data(self):
        return self.train_formatted

    def get_test_data(self):
        return self.test_formatted

    def get_train_label(self):
        return self.train_label

    def get_test_label(self):
        return self.test_label


class data_processor1m:
    def __init__(self, users_file='ml-1m/users.dat', genres_file='ml-100k/u.genre', movies_file='ml-1m/movies.dat',
                 ratings_file='ml-1m/ratings.dat'):
        self.genres = []
        self.genres_dict = {}
        self.users = [{}]
        self.movies = [{}]
        self.sex_dict = {'M': 1, 'F': 0}
        self.zipcode_dict = {}
        self.load_users(users_file)
        self.load_movies(genres_file, movies_file)
        self.matrix = [[0 for j in range(len(self.movies))] for i in range(len(self.users))]
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.load_data(ratings_file)
        # self.load_train_data(train_file)
        # self.load_test_data(test_file)

    def load_users(self, users_file):
        users_file = open(users_file, 'r')
        for line in users_file:
            line = line.strip().split('::')
            if line != []:
                line[1] = self.sex_dict[line[1]]
                line[2] = int(line[2])
                line[3] = int(line[3])
                try:
                    line[4] = self.zipcode_dict[line[4]]
                except Exception:
                    self.zipcode_dict[line[4]] = len(self.zipcode_dict)
                    line[4] = len(self.zipcode_dict)
                self.users.append({'sex': line[1], 'age': line[2], 'occupation': line[3], 'zip': line[4]})
        users_file.close()

    def load_genres(self, genres_file):
        genres_file = open(genres_file, 'r')
        for line in genres_file:
            line = line.strip()
            if line != '' and line != 'unknown|0':
                line = line.split('|')
                self.genres.append(line[0])
                self.genres_dict[line[0]] = int(line[1]) - 1
        genres_file.close()

    def load_movies(self, genres_file, movies_file):
        self.load_genres(genres_file)
        movies_file = open(movies_file, 'r', encoding='latin')
        for line in movies_file:
            line = line.strip().split('::')
            if line != []:
                while int(line[0]) != len(self.movies):
                    self.movies.append({})
                if '(' in line[1]:
                    line[1] = line[1].split('(')
                    if line[1][-1][-1] == ')':
                        year = int(line[1][-1][:-1])
                        line[1] = '('.join(line[1][:-1]).strip(' ')
                    else:
                        year = 1980
                        line[1] = '('.join(line[1])
                else:
                    year = 1980
                line[2] = line[2].split('|')
                genre = np.zeros(shape=[18])
                for g in line[2]:
                    genre[self.genres_dict[g]] = 1
                self.movies.append({'title': line[1], 'year': year, 'genre': genre})
        movies_file.close()
        self.bert_encode_titles()

    def bert_encode_titles(self):
        bc = BertClient(ip='hinton.cs.rutgers.edu')
        titles = list(map(lambda d: d['title'] if 'title' in d else 'None', self.movies[1:]))
        embeddings = bc.encode(titles)
        for i in range(1, len(self.movies)):
            self.movies[i]['title'] = embeddings[i - 1]

    def load_data(self, ratings_file):
        for i in range(len(self.users)):
            self.train_data.append([])
            self.train_label.append([])
            self.test_data.append([])
            self.test_label.append([])
        ratings_file = open(ratings_file, 'r')
        for line in ratings_file:
            line = line.strip()
            if line != '':
                line = line.split('::')
                user = int(line[0])
                movie = int(line[1])
                self.matrix[user][movie] = int(line[2])
                self.train_label[user].append(int(line[2]))
                line = {}
                for k, v in self.users[user].items():
                    line[k] = v
                for k, v in self.movies[movie].items():
                    line[k] = v
                self.train_data[user].append(line)

        for user, data in enumerate(self.train_data):
            if self.train_data[user] != []:
                self.train_data[user], self.test_data[user], self.train_label[user], self.test_label[
                    user] = train_test_split(self.train_data[user],
                                             self.train_label[user],
                                             test_size=0.2)

        cat = []
        for data in self.train_data:
            cat += data
        self.train_data = cat

        cat = []
        for data in self.train_label:
            cat += data
        self.train_label = cat

        cat = []
        for data in self.test_data:
            cat += data
        self.origin_test_x = self.test_data
        self.test_data = cat

        cat = []
        for data in self.test_label:
            cat += data
        self.origin_test_y = self.test_label
        self.test_label = cat

    def get_matrix(self):
        return self.matrix

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_label(self):
        return self.train_label

    def get_test_label(self):
        return self.test_label

    def save_data(self):
        pickle.dump(
            [self.train_data, self.test_data, self.train_label, self.test_label, self.matrix], open('data1m.pkl', 'wb'))
        pickle.dump([self.users, self.movies,
                     self.genres, self.genres_dict, self.zipcode_dict, self.sex_dict], open('data1m_add.pkl', 'wb'))
        pickle.dump([self.origin_test_x, self.origin_test_y], open('data1m_origin_test.pkl', 'wb'))


if __name__ == '__main__':
    dp = data_processor1m()
    dp.save_data()
    print('')
