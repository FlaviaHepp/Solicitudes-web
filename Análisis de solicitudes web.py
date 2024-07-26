""" El conjunto de datos contiene información sobre solicitudes web a un único sitio web. Es un conjunto de datos de series temporales, lo 
que significa que rastrea los datos a lo largo del tiempo. """

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, log_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import os
import random
from contextlib import contextmanager
from time import time
from tqdm import tqdm
import lightgbm as lgbm
import category_encoders as ce
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE


df=pd.read_csv('web_traffic.csv')

print(df)

print(df.shape)

print('fila es:' ,df.shape[0])
print('la columna es:' ,df.shape[1])

print(df.head())

print(df.info())

print(df.describe())

print(df.isna().sum())

print(df.duplicated().sum())

print(df.hist(color = "darkviolet", grid=False))

# Establece un título para el gráfico y etiqueta los ejes.
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include='number')

# Calcular la matriz de correlación
corr_matrix = numeric_df.corr()

# Crea un mapa de calor
sns.heatmap(corr_matrix, annot=True, cmap='inferno')
plt.show()

print(df)

df.groupby('Timestamp')['TrafficCount'].max()

head_tra=df.groupby('Timestamp')['TrafficCount'].sum().sort_values(ascending=False).head()
print(head_tra)

head_tra.plot(kind='bar')

#Tráfico Web Visualiza la importancia

#Preparación de datos
df = pd.read_csv("web_traffic.csv")
print(df[0:5])

print(df.columns.tolist())

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Year'] = df['Timestamp'].dt.year
df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
print(df['Hour'].value_counts())

plt.figure(figsize=(10, 6))
plt.plot(df['Timestamp'], df['TrafficCount'], marker='o', linestyle='-')
plt.xlabel('Marca de tiempo\n')
plt.ylabel('Recuento de tráfico\n')
plt.title('Recuento de tráfico a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df
data1=labelencoder(df)

#La fijacion de objetivos
target=['TrafficCount']
dataY=data1[target]
dataX=data1.drop(target+['Timestamp'],axis=1)
df_columns = list(dataX.columns)
print(df_columns)

m=len(dataX)
print(m)
M=list(range(m))
random.seed(2021)
random.shuffle(M)

trainX=dataX.iloc[M[0:(m//5)*4]]
trainY=dataY.iloc[M[0:(m//5)*4]]
testX=dataX.iloc[M[(m//5)*4:]]
testY=dataY.iloc[M[(m//5)*4:]]

train_df=trainX
test_df=testX
train_df.columns=df_columns
test_df.columns=df_columns
def create_numeric_feature(input_df):
    use_columns = df_columns 
    return input_df[use_columns].copy()


class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def to_feature(input_df):

    processors = [
        create_numeric_feature,
    ]
    
    out_df = pd.DataFrame()
    
    for func in tqdm(processors, total=len(processors)):
        with Timer(prefix='create' + func.__name__ + ' '):
            _df = func(input_df)

        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df
train_feat_df = to_feature(train_df)
test_feat_df = to_feature(test_df)

#Modelo
def fit_lgbm(X, y, cv, 
             params: dict=None):

    if params is None:
        params = {}

    models = []
    oof_pred = np.zeros_like(y, dtype=float)

    for i, (idx_train, idx_valid) in enumerate(cv): 
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)
        
        with Timer(prefix='fit fold={} '.format(i)):
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)])

        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        print(f'Doblar {i} RMSLE: {mean_squared_error(y_valid, pred_i) ** .5:.4f}')
        print()

    score = mean_squared_error(y, oof_pred) ** .5
    print('-' * 50)
    print('TERMINADO | RMSLE entero: {:.4f}'.format(score))
    return oof_pred, models
params = {
    'objective': 'rmse', 
    'learning_rate': .1,
    'reg_lambda': 1.,
    'reg_alpha': .1,
    'max_depth': 5, 
    'n_estimators': 10000, 
    'colsample_bytree': .5, 
    'min_child_samples': 10,
    'subsample_freq': 3,
    'subsample': .9,
    'importance_type': 'gain', 
    'random_state': 71,
    'num_leaves': 62
}
y = trainY
ydf=pd.DataFrame(y)
ydf

for i in range(1):
    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    ydfi=ydf.iloc[:,i]
    y=np.array(ydfi)
    cv = list(fold.split(train_feat_df, y))
    oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params)
    
    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_title(target[i],fontsize=16)
    ax.set_ylabel('predicho\n',fontsize=12)
    ax.set_xlabel('actual\n',fontsize=12)
    ax.scatter(y,oof,alpha=0.2)
    
#Visualizar importancia
def visualize_importance(models, feat_train_df):

    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x='feature_importance', 
                  y='column', 
                  order=order, 
                  ax=ax, 
                  palette='inferno', 
                  orient='h')
    
    ax.tick_params(axis='x', rotation=0)
    #ax.set_title('Importancia')
    ax.grid()
    fig.tight_layout()
    
    return fig,ax

#fig, ax = visualize_importance(models, train_feat_df)
for i in range(1):
    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    ydfi=ydf.iloc[:,i]
    y=np.array(ydfi)
    cv = list(fold.split(train_feat_df, y))
    oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params)
    
fig, ax = visualize_importance(models, train_feat_df)
ax.set_title(target[i]+' Importancia\n',fontsize=20)

preds=[]
for i in range(5):
    preds += [models[i].predict(test_feat_df.values)/5]
predsT=np.array(preds).T
preds2=[]
for item in predsT:
    value=sum(item)
    preds2+=[value]
print(preds2[0:5])

for i in range(1):
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(oof, label='Tren previsto '+target[i], ax=ax, color='C1',bins=30)
    sns.histplot(preds2, label='Prueba prevista '+target[i], ax=ax, color='fuchsia',bins=30)
    ax.legend()
    ax.grid()
    
fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Prueba real frente a prueba prevista\n', fontsize = '16', fontweight = 'bold')
ax.set_ylabel('Prueba prevista\n',fontsize=12)
ax.set_xlabel('Prueba real\n',fontsize=12)
ax.scatter(testY,preds2,alpha=0.2)
plt.show()

fig,ax = plt.subplots(figsize=(6,6))
ax.set_title('Hora frente a número de tráfico\n', fontsize = '16', fontweight = 'bold')
ax.set_ylabel('Recuento de tráfico\n',fontsize=12)
ax.set_xlabel('Hora\n',fontsize=12)
ax.scatter(df['Hour'],df['TrafficCount'],alpha=0.2)
plt.show()
