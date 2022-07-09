# -*- coding: utf-8 -*-
import os
import pandas as pd 
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import json
from math import sqrt

seed = 42
suf = 2

def get_sleep_duration(tobed: str, wakeup: str):
    h,m,s = tobed.split(':')
    t0 = int(h) + int(m)/60
    h,m,s = wakeup.split(':')
    t1 = int(h) + int(m)/60
    t = t1 - t0
    if t < 0:
        t += 24.
    return t

def get_pred(probs, th=0.5):
    y_pred = [0] * len(probs)
    for i, p in enumerate(probs):
        if p > th:
            y_pred[i] = 1
    return y_pred

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
  ret = {}                          # Создаём пустой словарь
  for _id, name in enumerate(s):    # Проходим по всем парам - id и название
    ret.update({name: _id})         # Добавляем в словарь
  return ret

mul_features = ['Пол', 'Семья', 'Этнос', #'Национальность', 'Религия',
       'Образование', #'Профессия',
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
ND = len(diseases)

dic = {}
for feature in mul_features:
    dic[feature] = create_dict(set(data_df[feature]))

y_data = []
for i in range(ND):
    y_data.append(data_df[diseases[i]].values)
    # print(y_data[i].mean())

ths = np.mean(y_data, axis=1)
print(ths)

x_test = []
for _id, person in enumerate(np.array(test_df)):
    x_tr = []
    for feature in mul_features:
        arr = np.zeros(len(dic[feature]), np.float64)
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

avg_recall = 1e-4

"""## Предсказание на тесте"""

y_preds = [None] * ND

for d in range(ND):
    model = tf.keras.models.load_model(f'models/{seed}-{d}.h5')
    print(f"loaded from models/{seed}-{d}.h5")
    test_probs = model.predict(x_test)
    y_preds[d] = get_pred(test_probs, ths[d])

res_df = pd.DataFrame()
res_df['ID'] = test_df.ID.values
for d in range(ND):
    res_df[diseases[d]] = y_preds[d]
# print(res_df)

name = 'submit-%04d-%d-%d.csv' % (int(avg_recall * 10000), seed, suf)
print(name)
res_df.to_csv(name, index=False)
