# Pacotes e Definindo a semente
## Manipulação e visualização de dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Modelagem
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

## Setando a semente
seed = 2021

# Importando dados
data = pd.read_csv('https://raw.githubusercontent.com/mascalmeida/colab_classes/main/datasets/water_potability.csv')
data.head()

# Identificando problemas
## Valores faltantes
nas = pd.DataFrame(data.isna().sum()).reset_index().rename(columns={"index": "variavel", 0: "nas"})
nas['nas_100'] = 100 - ((abs(len(data) - nas['nas'])/len(data))*100)
print('Quantidade de valores faltantes')
print(nas)
## Dados desbalanceados
print('\n\n')
print('Balanceamento dos dados')
print(data['Potability'].value_counts())

# Lidando com os valores faltantes considerando o desbalanceamento
data_novo = data.loc[~((data['Potability'] == 0) & ((data['ph'].isna()) | (data['Sulfate'].isna()) | (data['Trihalomethanes'].isna()))), :].\
                  dropna(subset=['Trihalomethanes']).\
                  interpolate(method='linear', limit_direction='backward').\
                  dropna()
# Conferindo
## Valores faltantes
nas = pd.DataFrame(data_novo.isna().sum()).reset_index().rename(columns={"index": "variavel", 0: "nas"})
nas['nas_100'] = 100 - ((abs(len(data_novo) - nas['nas'])/len(data_novo))*100)
print('Quantidade de valores faltantes')
print(nas)
## Dados desbalanceados
print('\n\n')
print('Balanceamento dos dados')
print(data_novo['Potability'].value_counts())

# Pré processamento
## Reservando dados para validação do modelo
data_modelo, test = train_test_split(data_novo, 
                                     stratify=data_novo['Potability'], 
                                     random_state = seed, 
                                     test_size = 0.1)
## Definindo entradas e saída
### Saída
y = data_modelo['Potability']
### Entradas
x = data_modelo.drop(columns=['Potability'])

# Pré processamento 2
## Dividindo o dataset em treino e validação
x_train, x_valid, y_train, y_valid = train_test_split(x, y, 
                                                      stratify=y, 
                                                      random_state = seed, 
                                                      test_size = 0.20)

# Treinando o modelo (árvore de decisão)
modelo = DecisionTreeClassifier(random_state = seed)
modelo = modelo.fit(x_train, y_train)
modelo1 = modelo
# Predição e avaliação do modelo
y_valid_pred = modelo.predict(x_valid)
## print classification report
print('Avaliando o modelo -----------------------\n', classification_report(y_valid, y_valid_pred))
##Confusion Matrix
plot_confusion_matrix(modelo, x_valid, y_valid, normalize='true', cmap='binary')
plt.show();

# Modelagem
from sklearn.model_selection import GridSearchCV
# Importando o Make Scorer
from sklearn.metrics import make_scorer
# Importando os módulos de cálculo de métricas
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

# Criando um dicionário com as métricas que desejo calcular.
meus_scores = {'accuracy' :make_scorer(accuracy_score),
               'recall'   :make_scorer(recall_score),
               'precision':make_scorer(precision_score),
               'f1'       :make_scorer(fbeta_score, beta = 1)}


# Definindo o 'K' para o K-Fold
k = 5
## Defining parameter range to grid search
param_grid = [{'splitter': ['best', 'random']}]
# Treinando o modelo (árvore de decisão)
clf = DecisionTreeClassifier(random_state = seed)
modelo = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      scoring = meus_scores,
                      refit = 'f1',
                      cv=k)
modelo.fit(x, y);
modelo2 = modelo
# Results
print('Model =', modelo.best_params_)
display(pd.DataFrame(modelo.cv_results_)[['params', 'mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_f1']])

# Criando um dicionário com as métricas que desejo calcular.
meus_scores = {'accuracy' :make_scorer(accuracy_score),
               'recall'   :make_scorer(recall_score),
               'precision':make_scorer(precision_score),
               'f1'       :make_scorer(fbeta_score, beta = 1)}


# Definindo o 'K' para o K-Fold
k = 10
## Defining parameter range to grid search
param_grid = [{'splitter': ['best', 'random']}]
# Treinando o modelo (árvore de decisão)
clf = DecisionTreeClassifier(random_state = seed)
modelo = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      scoring = meus_scores,
                      refit = 'f1',
                      cv=k)
modelo.fit(x, y);
modelo10 = modelo
# Results
print('Model =', modelo.best_params_)
display(pd.DataFrame(modelo.cv_results_)[['params', 'mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_f1']])

print(y.value_counts())

# Criando um dicionário com as métricas que desejo calcular.
meus_scores = {'accuracy' :make_scorer(accuracy_score),
               'recall'   :make_scorer(recall_score),
               'precision':make_scorer(precision_score),
               'f1'       :make_scorer(fbeta_score, beta = 1)}


# Definindo o 'K' para o K-Fold
k = 1080
## Defining parameter range to grid search
param_grid = [{'splitter': ['best', 'random']}]
# Treinando o modelo (árvore de decisão)
clf = DecisionTreeClassifier(random_state = seed)
modelo = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      scoring = meus_scores,
                      refit = 'f1',
                      cv=k)
modelo.fit(x, y);
LOOCV = modelo
# Results
print('Model =', modelo.best_params_)
display(pd.DataFrame(modelo.cv_results_)[['params', 'mean_test_accuracy','mean_test_precision','mean_test_recall','mean_test_f1']])

## Definindo entradas e saída
### Saída
y_test = test['Potability']
### Entradas
x_test = test.drop(columns=['Potability'])

# Predição e avaliação do modelo 1
y1 = modelo1.predict(x_test)
## print classification report
print('Avaliando o modelo 1 -----------------------\n', classification_report(y_test, y1))
##Confusion Matrix
plot_confusion_matrix(modelo1, x_test, y_test, normalize='true', cmap='binary')
plt.show();

# Predição e avaliação do modelo 2
y2 = modelo2.predict(x_test)
## print classification report
print('Avaliando o modelo 2 -----------------------\n', classification_report(y_test, y2))
##Confusion Matrix
plot_confusion_matrix(modelo2, x_test, y_test, normalize='true', cmap='binary')
plt.show();

# Predição e avaliação do modelo 10
y10 = modelo10.predict(x_test)
## print classification report
print('Avaliando o modelo 10 -----------------------\n', classification_report(y_test, y10))
##Confusion Matrix
plot_confusion_matrix(modelo10, x_test, y_test, normalize='true', cmap='binary')
plt.show();

# Predição e avaliação do modelo LOOCV
yLOOCV = LOOCV.predict(x_test)
## print classification report
print('Avaliando o LOOCV -----------------------\n', classification_report(y_test, yLOOCV))
##Confusion Matrix
plot_confusion_matrix(LOOCV, x_test, y_test, normalize=None, cmap='binary')
plt.show();
