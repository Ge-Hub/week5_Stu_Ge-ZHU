import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics  as metrics
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib
import csv


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat
from deepctr.feature_column import get_feature_names


print("import ready")

#数据集分为三个文件：用户数据users.dat，电影数据movies.dat和评分数据ratings.dat。
### 用户数据:分别有用户ID、性别、年龄、职业ID和邮编等字段。

users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table('users.dat', sep='::', header=None, names=users_title, engine = 'python')
#print(users.head)

### 电影数据: 分别有电影ID、电影名和电影风格等字段。数据中的格式：MovieID::Title::Genres
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table('movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
#print(movies.head)

### 评分数据: 分别有用户ID、电影ID、评分和时间戳等字段。评分字段Rating就是我们要学习的targets，时间戳字段我们不使用。
ratings_title = ['UserID','MovieID', 'Rating', 'timestamps']
ratings = pd.read_table('ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
#print(ratings.head)


data1= pd.merge(pd.merge(ratings,users),movies)
print(data1[:10])

#df1 = pd.merge(ratings, users)
#print(df1[:200])
#df2 = pd.merge(df1,movies)
#print(df2[:200])


csv_file = open('deepfm_movielens_full.csv','w',newline='',encoding='utf-8')
#调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='utf-8'。
writer = csv.writer(csv_file)
# 用csv.writer()函数创建一个writer对象。
writer.writerow(data1)
movielens_full = []
for i in range (0, data1.shape[0]):
    temp = []
    for j in range (0, data1.shape[1]):
        if str(data1.values[i,j] != 'nan'):
            temp.append (str(data1.values[i,j]))

movielens_full.append(temp)
print(movielens_full[:20])
csv_file.close()
#写入完成后，关闭文件就大功告成啦！



print("-"*188)

#数据加载
data = pd.read_csv("deepfm_movielens_full.csv")
sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['rating']


# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
print(fixlen_feature_columns)
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5

print("-"*118)
print("test MES:", mse)
print("-"*118)
print("test RMSE:", rmse)