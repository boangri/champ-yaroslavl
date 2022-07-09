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

save = True
suf = 0
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)


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
    h,m,s = tobed.split(':')
    t0 = int(h) + int(m)/60
    h,m,s = wakeup.split(':')
    t1 = int(h) + int(m)/60
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

x_train = np.array(x_data, dtype=np.float32)
y_train = []
for i in range(ND):
    y_train.append(y_data[i])

print(x_train.shape, y_train[0].shape)

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

"""## Подбор гиперпараметров"""


def get_pred(probs, th=0.5):
    y_pred = [0] * len(probs)
    for i, p in enumerate(probs):
        if p > th:
            y_pred[i] = 1
    return y_pred


def build_model(config):
    model = Sequential()

    model.add(Dense(config['latent_dim'], activation='relu'))
    model.add(Dropout(config['dropout']))

    if config['num_layers'] > 1:
        model.add(Dense(config['latent_dim2'], activation='relu'))
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
        metrics=[]
    )

    return model


if save:
    filename = "test-1200-best-params.json"
    with open(filename, 'r') as f:
        data = json.load(f)
    all_best_params = json.loads(data)
    # print(json.dumps(all_best_params, indent=2, sort_keys=True))
    for d in range(ND):
        params = all_best_params[d]
        num_epochs = params['num_epochs']
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        model = build_model(params)
        history = model.fit(x_data, y_data[d], batch_size=len(x_data), epochs=num_epochs, verbose=0)
        model.save(f'models/{seed}-{d}.h5')
        print(f"saved to models/{seed}-{d}.h5")

"""Кросс Валидация - оценка результата"""


def evaluate_model(model, X, y, num_folds, th):
    cv_scores = []
    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
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
# var = 0
# for d in range(ND):
#     model = tf.keras.models.load_model(f'models/{seed}-{d}.h5')
#     print(f"loaded from models/{seed}-{d}.h5")
#     # num_epochs = all_best_params[d]['num_epochs']
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     recall, std = evaluate_model(model, X=x_data, y=y_data[d], num_folds=5, th=ths[d])
#     avg_recall += recall
#     var += std*std
#     print("recall%d=%.4f std=%.4f" % (d, recall, std))
# avg_recall /= ND
# var /= (ND - 1)
# print("Estimate Recall=%.4f std=%.4f" %(avg_recall, sqrt(var)))

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
