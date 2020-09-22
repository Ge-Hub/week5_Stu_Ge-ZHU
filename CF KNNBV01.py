
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
#cross_validate() 是被调用的外部接口，fit_and_score() 是在 cross_validate() 中被调用的。
    #输入有算法对象，数据集，需要测量的指标，交叉验证的次数等。它对输入的数据 data，分成 cv 份，然后每次选择其中一份作为测试集，其余的作为训练集。在数据集划分完后，对它们分别调用 fit_and_score()，去进行算法拟合。
    #对数据集的划分不是静态全部划分完，然后分别在数据集上进行训练和验证，而是利用输入的 data 构造一个生成器，每次抛出一组划分完的结果。
    #对 fit_and_score() 函数，它对输入的算法在输入的训练集上进行拟合，然后在输入的测试集上进行验证，再计算需要的指标。
from surprise.model_selection import cross_validate
import pandas as pd
news = pd.read_csv('ratings.csv')
print(news.head)

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
train_set = data.build_full_trainset()


from surprise import KNNWithZScore
algo=KNNWithZScore(k=50, sim_options={'user_based': False, 'verbose': 'True'})
algo.fit(train_set)
uid=str(196)
iid=str(332)
pred=algo.predict(uid,iid,r_ui=4,verbose=True)

kf=KFold(n_splits=3)
for trainset,testset in kf.split(data):
    algo.fit(trainset)
    predictions=algo.test(testset)
    #计算RMSE,AME
    accuracy.rmse(predictions,verbose=True)
    accuracy.mae(predictions,verbose=True)


### 使用协同过滤正态分布 User based
from surprise import KNNWithZScore
algo = KNNWithZScore(k=50, sim_options={'user_based': False, 'verbose': 'True'})
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print("KNNWithZScore Results:",perf)
print("-"*118)

### 使用协同过滤正态分布 Item based
from surprise import KNNWithZScore
algo = KNNWithZScore(k=50, sim_options={'user_based': True, 'verbose': 'True'})
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print("KNNWithZScore Results:",perf)
print("-"*118)

### 使用基础版协同过滤
from surprise import KNNBasic
algo = KNNBasic(k=50, sim_options={'user_based': False, 'verbose': 'True'})
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print("KNNBasic Results:",perf)
print("-"*118)

### 使用均值协同过滤
from surprise import KNNWithMeans
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print("KNNWithMeans Results:",perf)
print("-"*118)

### 使用协同过滤baseline
from surprise import KNNBaseline
algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})
perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
print("KNNBaseline Results:",perf)
print("-"*118)
