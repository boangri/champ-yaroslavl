# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, Adadelta

from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import KFold
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt import space_eval
import json
from math import sqrt
from time import time

N_BOTS = 40  # размер популяции
N_GEN = 5 # число поколений
K_FOLDS = 5
jsonfile = 'configs-%d-%d.json' % (N_BOTS, N_GEN)
seed = 1703
dis = list(range(5))
ND = len(dis)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_sleep_duration(tobed: str, wakeup: str):
    h, m, s = tobed.split(':')
    t0 = int(h) + int(m) / 60
    h, m, s = wakeup.split(':')
    t1 = int(h) + int(m) / 60
    t = t1 - t0
    if t < 0:
        t += 24.
    return t


data_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_dataset_test.csv')

data_df['Сигарет в день'] = data_df['Сигарет в день'].fillna(0)
test_df['Сигарет в день'] = test_df['Сигарет в день'].fillna(0)
test_df['Статус Курения'].iloc[49] = 'Никогда не курил(а)'

tobed = data_df['Время засыпания'].values
wakeup = data_df['Время пробуждения'].values

durs = []
for i in range(len(tobed)):
    t = get_sleep_duration(tobed[i], wakeup[i])
    # print(t)
    durs.append(t)
AVG, STD = np.mean(durs), np.std(durs)


# Создаём словарь поле - его индекс
def create_dict(s):
    ret = {}  # Создаём пустой словарь
    for _id, name in enumerate(s):  # Проходим по всем парам - id и название
        ret.update({name: _id})  # Добавляем в словарь
    return ret


mul_features = ['Пол', 'Семья', 'Этнос',  # 'Национальность', 'Религия',
                'Образование',  # 'Профессия',
                'Статус Курения',
                'Частота пасс кур', 'Алкоголь']
#    'Время засыпания', 'Время пробуждения']
#    'Возраст курения', 'Возраст алког',
bin_features = ['Вы работаете?', 'Выход на пенсию',
                'Прекращение работы по болезни', 'Сахарный диабет', 'Гепатит',
                'Онкология', 'Хроническое заболевание легких', 'Бронжиальная астма',
                'Туберкулез легких ', 'ВИЧ/СПИД',
                'Регулярный прим лекарственных средств', 'Травмы за год', 'Переломы',
                'Пассивное курение',
                'Сон после обеда',
                'Спорт, клубы', 'Религия, клубы']

diseases = ['Артериальная гипертензия',
            'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда',
            'Сердечная недостаточность', 'Прочие заболевания сердца']
# ND = len(diseases)

dic = {}
for feature in mul_features:
    dic[feature] = create_dict(set(data_df[feature]))

y_data = []
for i in range(ND):
    y_data.append(data_df[diseases[i]].values)
    print(y_data[i].mean())

x_data = []

for _id, person in enumerate(np.array(data_df)):
    x_tr = []
    for feature in mul_features:
        arr = np.zeros(len(dic[feature]), np.float32)
        arr[dic[feature][data_df.loc[_id, feature]]] = 1.
        x_tr.extend(arr)

    for feature in bin_features:
        x_tr.append(data_df.loc[_id, feature])

    x_tr.append(data_df.loc[_id, 'Сигарет в день'] / 10.)

    tobed = data_df.loc[_id, 'Время засыпания']
    wakeup = data_df.loc[_id, 'Время пробуждения']
    x_tr.append((get_sleep_duration(tobed, wakeup) - AVG) / STD)

    x_data.append(x_tr)

x_data = np.array(x_data, dtype=np.float32)
x_train = x_data[:]
# x_val = x_data[N:]
y_train = []
# y_val = []
for i in range(ND):
    y_train.append(y_data[i])
    # y_val.append(y_data[i][N:])

print(x_train.shape, y_train[0].shape)  # , x_val.shape, y_val[4].shape)

x_test = []

for _id, person in enumerate(np.array(test_df)):
    x_tr = []
    # print(_id)
    for feature in mul_features:
        arr = np.zeros(len(dic[feature]), np.float64)

        # print(feature, test_df.loc[_id, feature])
        arr[dic[feature][test_df.loc[_id, feature]]] = 1.
        x_tr.extend(arr)

    for feature in bin_features:
        x_tr.append(test_df.loc[_id, feature])

    x_tr.append(test_df.loc[_id, 'Сигарет в день'] / 10.)

    tobed = test_df.loc[_id, 'Время засыпания']
    wakeup = test_df.loc[_id, 'Время пробуждения']
    x_tr.append((get_sleep_duration(tobed, wakeup) - AVG) / STD)

    x_test.append(x_tr)

x_test = np.array(x_test, dtype=np.float64)
print(x_test.shape)

"""## Подбор гиперпараметров"""


def get_pred(probs, th=0.5):
    # y_pred = np.zeros(len(probs), dtype=np.float32)
    y_pred = [0] * len(probs)
    for i, p in enumerate(probs):
        if p > th:
            y_pred[i] = 1
    # print("get_pred: avg=%.4f std=%.4f max=%.4f min=%.4f" % (probs.mean(), probs.std(), probs.max(), probs.min()))
    return y_pred


ths = np.mean(y_data, axis=1)
print(ths)


class Bot:
    NG = 7 # number of gens

    def __init__(self, d):
        self.d = d
        self.loss = None
        self.loss_std = None
        self.gen = [0.5] * self.NG

    def init(self):
        self.gen = [random.random() for _ in range(self.NG)]
        self.calc_loss()

    def get_config(self):
        opts = ['adagrad', 'adam', 'nadam', 'rmsprop']
        config = {
            'd': self.d,
            'loss': self.loss,
            'loss_std': self.loss_std,
            'dropout': self.gen[0],
            'latent_dim': int(pow(2, 6 + 5 * self.gen[1])),
            'latent_dim2': int(pow(2, 6 + 5 * self.gen[2])),
            'lr': pow(10, -(2 + 2 * self.gen[3])),
            'num_epochs': int(20 + 40 * self.gen[4]),
            'num_layers': 1 if self.gen[5] < 0.5 else 2,
            'opt': opts[int(4 * self.gen[6] - 1e-6)],
        }
        return config

    def calc_loss(self):
        config = self.get_config()
        result = objective(config)
        self.loss = result['loss']
        self.loss_std = result['std']
        print('.', end='')

    def mutation(self, p):
        for i, g in enumerate(self.gen):
            x = g + p * random.random()
            if x > 1:
                x = 1. - 1e-6
            if x < 0:
                x = 1e-6
            self.gen[i] = x


class GA:
    def __init__(self, n, d, split=0.3, reverse=False):
        """
        n: число ботов (размер популяции)
        l: размерность бота
        split: доля популяции, переходящая в новое поколение
        reverse: флаг True - ищем максимум функции потерь
        dtype: тип данных бота (float, int)
        """
        self.n = n
        self.d = d
        self.nsurv = int(n * split)
        self.popul = [Bot(self.d) for x in range(self.n)]
        self.reverse = reverse
        for bot in self.popul:
            bot.init()
        print('initialization')
        self.sort_bots()

    def sort_bots(self):
        self.popul = sorted(self.popul, key=lambda x: x.loss, reverse=self.reverse)

    def new_generation(self):
        for i in range(self.nsurv, self.n):
            i1, i2 = np.random.choice(self.nsurv, 2, replace=False)  # выбрали 2-х родителей
            bot = self.child(i1, i2)
            self.popul[i] = bot

    def mutation(self):
        for bot in self.popul[self.nsurv:]:
            bot.mutation(0.1)
            bot.calc_loss()
        print('mutation')

    def evolution(self):
        self.new_generation()
        self.mutation()
        self.sort_bots()

    def child(self, i1, i2):
        bot = Bot(self.d)
        for j in range(Bot.NG):
            bot.gen[j] = self.popul[i1].gen[j] if np.random.rand() < 0.5 else self.popul[i2].gen[j]
        return bot


def build_model(config):
    model = Sequential()

    model.add(Dense(config['latent_dim'], activation='relu'))
    # if config['batch_norm'] > 0:
    #     model.add(BatchNormalization())
    model.add(Dropout(config['dropout']))

    if config['num_layers'] > 1:
        model.add(Dense(config['latent_dim2'], activation='relu'))
        # if config['batch_norm'] > 0:
        #     model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))

    model.add(Dense(1, activation='sigmoid'))

    if config['opt'] == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=config['lr'])
    if config['opt'] == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=config['lr'])
    if config['opt'] == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    if config['opt'] == 'ftlr':
        opt = tf.keras.optimizers.Ftrl(learning_rate=config['lr'])
    if config['opt'] == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=config['lr'])
    if config['opt'] == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=config['lr'])
    if config['opt'] == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=config['lr'])
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[
            #   tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            #   tf.keras.metrics.Recall(name='recall'),
            #   tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


# оптимизируем val_loss
def cross_validation(config, X, y, num_folds):
    cv_scores = []
    num_epochs = config['num_epochs']
    kf = KFold(n_splits=num_folds, random_state=None, shuffle=True)
    for train_index, val_index in kf.split(X):
        x_train = X[train_index]
        y_train = y[train_index]
        x_val = X[val_index]
        y_val = y[val_index]
        # tf.random.set_seed(seed)
        model = build_model(config)
        hist = model.fit(
            x_train, y_train,
            epochs=num_epochs,
            batch_size=len(x_train),
            validation_data=(x_val, y_val),
            verbose=0
        )
        # val_probs = model.predict(x_val)
        # y_pred = get_pred(val_probs, config['th'])
        loss = hist.history['val_loss'][-1]
        cv_scores.append(loss)
    return cv_scores


def objective(params):
    d = params['d']
    # print(f"d={d} Selected model hyperparams:")
    # print(params)
    cv = cross_validation(
        config=params,
        X=x_data,
        y=y_data[d],
        num_folds=K_FOLDS,
    )
    mean_cv = np.mean(cv)
    std_cv = np.std(cv)
    # print('Avg loss: %.4f std: %.4f' % (mean_cv, std_cv))
    return {'loss': mean_cv, 'status': STATUS_OK, 'std': std_cv, 'params': params}


ths = np.mean(y_data, axis=1)
best_configs = [None] * ND

for d in dis:
    t0 = time()
    ga = GA(N_BOTS, d)
    bot = ga.popul[0]
    print("d=%d generation %d loss=%.4f time=%.1f" % (d, 0, bot.loss, time() - t0))
    for i in range(N_GEN):
        t0 = time()
        ga.evolution()
        bot = ga.popul[0]
        print("d=%d generation %d loss=%.4f time=%.1f" % (d, i+1, bot.loss, time() - t0))
    print("d=%d Best config:" % d)
    config = bot.get_config()
    print(config)
    best_configs[d] = config

json_string = json.dumps(best_configs, cls=NpEncoder)
with open(jsonfile, 'w') as outfile:
    json.dump(json_string, outfile)

models = []

for d in dis:
    params = best_configs[d]
    num_epochs = params['num_epochs']
    tf.random.set_seed(seed)
    model = build_model(params)
    history = model.fit(x_data, y_data[d], batch_size=len(x_data), epochs=num_epochs, verbose=0)
    models.append(model)

"""## Предсказание на тесте"""

ths = np.mean(y_data, axis=1)
y_preds = [None] * ND

for d in dis:
    test_probs = models[d].predict(x_test)
    y_preds[d] = get_pred(test_probs, ths[d])

res_df = pd.DataFrame()
res_df['ID'] = test_df.ID.values
for d in dis:
    res_df[diseases[d]] = y_preds[d]
print(res_df)

name = 'ga-%d-%d-%d.csv' % (N_BOTS, N_GEN, seed)
print(name)
res_df.to_csv(name, index=False)

"""Кросс Валидация - оценка результата"""


def evaluate_model(model, X, y, num_folds, th):
    cv_scores = []
    kf = KFold(n_splits=num_folds, random_state=None, shuffle=True)
    for train_index, val_index in kf.split(X):
        x_val = x_data[val_index]
        y_val = y[val_index]
        probs = model.predict(x_val)
        y_pred = get_pred(probs, th)
        recall = recall_score(y_val, y_pred, average='macro')
        cv_scores.append(recall)
    avg_recall = np.mean(cv_scores)
    std_recall = np.std(cv_scores)
    return avg_recall, std_recall


avg_recall = 0
var = 0
for d in dis:
    num_epochs = best_configs[d]['num_epochs']
    recall, std = evaluate_model(models[d], X=x_data, y=y_data[d], num_folds=5, th=ths[d])
    avg_recall += recall
    var += std * std
    print("recall%d=%.4f std=%.4f" % (d, recall, std))
avg_recall /= ND
var /= ND
print("Estimate Recall=%.4f std=%.4f" % (avg_recall, sqrt(var)))
