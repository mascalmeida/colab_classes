# Pacotes

## Dataset
from sklearn.datasets import load_iris

# STRUCTURAL PACKAGES ----------------------------------------------------------
## Manipulação de dados
import pandas as pd
## Operate data
import numpy as np
## random package for set seed function
import random

# VISUALIZATION ----------------------------------------------------------------
## Categorical data
import seaborn as sns
## Numerical data
import matplotlib.pyplot as plt

# DATA SCIENCE -----------------------------------------------------------------
## Scale a data frame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
## Split dataset to train and test
from sklearn.model_selection import train_test_split
## Classification Models
from sklearn.ensemble import RandomForestClassifier
## Tuning parameter with croos-validation (k-fold)
from sklearn.model_selection import GridSearchCV
## Metrics
from sklearn.metrics import classification_report, confusion_matrix
#Install -> !pip install scikit-plot
!pip install scikit-plot
import scikitplot as skplt

# Garantindo a reprodutibilidade
## Set seed function
seed = 1000
print('Seed =', seed)

# Importando dados
## Carregando dados iris
iris = load_iris()
## Transformando em um dataframe pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names)
## Adicionando uma variável
df['target']=iris.target
## Criando uma variável
df['target_name'] = iris.target_names[2]
df.loc[df['target'] == 0, 'target_name'] = iris.target_names[0]
df.loc[df['target'] == 1, 'target_name'] = iris.target_names[1]
## Exibindo o dataframe
display(df)

# Definindo output
## Preservando o dataset original
dt = df.copy()
## Visualizando quantidade de classes
print(dt['target_name'].value_counts())

## Criando variável - Setosa
dt['setosa'] = 0
## Adicionando os eventos a variável - Setosa
dt.loc[dt['target_name'] == 'setosa', 'setosa'] = 1

## Criando variável - Versicolor
dt['versicolor'] = 0
## Adicionando os eventos a variável - Versicolor
dt.loc[dt['target_name'] == 'versicolor', 'versicolor'] = 1

## Criando variável - Virginica
dt['virginica'] = 0
## Adicionando os eventos a variável - Virginica
dt.loc[dt['target_name'] == 'virginica', 'virginica'] = 1

## Exibindo o dataframe
display(dt)

# Separando entradas e saídas
## Output
y = dt['setosa']
## Inputs
x = dt.loc[:, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
## Exibindo
display(y, x)

# Dividindo dataset em treino e teste
## Split training and test dataset
x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x, 
                                                                      y, 
                                                                      stratify=y,
                                                                      random_state = seed,
                                                                      test_size = 0.25)

# Scale dataset
## Fit based on data train
scaler_fit = MinMaxScaler().fit(x_train_unscaled)
## Transform data train and data test
x_train = pd.DataFrame(scaler_fit.transform(x_train_unscaled), 
                       columns=x_train_unscaled.columns, 
                       index=x_train_unscaled.index)
x_test = pd.DataFrame(scaler_fit.transform(x_test_unscaled), 
                      columns=x_test_unscaled.columns,
                      index=x_test_unscaled.index)
## Exibindo dados
display(x_train, x_test.head())

# Grid Search Cross-Validation
## Define 'K' for K-Fold
k = 5
## Defining parameter range to grid search
param_grid = [{'n_estimators': [2000],
               'bootstrap': [True, False]}]
## Define method instance  
clf = RandomForestClassifier(random_state=seed)

## Define grid instance
grid = GridSearchCV(estimator=clf, 
                    param_grid=param_grid, 
                    cv=k,
                    scoring='f1')
## Initialize grid search, fitting the best model
grid.fit(x_train, y_train); 

# Results
print('Model =', grid.best_params_)

## Make predictions over test set for both models
pred = grid.predict(x_test)
## print classification report
print('MODEL -------------------------------------------------------\n', 
      classification_report(y_test, pred))
##Confusion Matrix
skplt.metrics.plot_confusion_matrix(y_test, pred, normalize=False, title='Confusion Matrix')
plt.tight_layout()
plt.show();
