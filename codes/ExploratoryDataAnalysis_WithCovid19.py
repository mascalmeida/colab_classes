# Pacotes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando dados
## Link do raw data do github
link = 'https://github.com/mascalmeida/colab_classes/blob/main/owid-covid-data.xlsx?raw=true'
## Carregando o dataset
dt = pd.read_excel(link)
## Visualizando o cabeçalho do dataset
dt.head()

# Seleção de variáveis
## Informações gerais das variáveis
display(dt.info())
## Selecionando variáveis para investigação
df = dt.loc[:, ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'stringency_index', 'population']].dropna().reset_index(drop=True)
display(df.head())

# Filtrando os dados
## Visualizando categorias
print(df['continent'].value_counts())
## Aplicando diferentes filtros
df_fil = df.loc[(df['continent'] == 'South America') & 
                (df['population'] > 21000000) &
                (df['date'] > '2021-01-01'), :].reset_index(drop=True)
display(df_fil)

# Criando variáveis
## Preservando o dataset
df_new = df_fil.copy()
## Criando taxa de mortos/casos
df_new['rate_deaths_cases'] = (df_new['total_deaths']/df_new['total_cases'])
## Criando variável categorica
df_new['pop'] = 'Pequena'
df_new.loc[(df_new['population'] >= 10000000) & (df_new['population'] < 40000000), 'pop'] = 'Media'
df_new.loc[df_new['population'] >= 40000000, 'pop'] = 'Grande'
## Exibindo dados finais
display(df_new)

# Análise descritiva dos dados
print('Descritiva total:')
display(df_new.describe())
print('Descritiva do Brasil:')
display(df_new.loc[df_new['iso_code'] == 'BRA', :].describe())
print('Média da taxa de mortos por casos de Brasil x Argentina:')
print('Brasil = ', df_new.loc[df_new['iso_code'] == 'BRA', 'rate_deaths_cases'].mean())
print('Argentina = ', df_new.loc[df_new['iso_code'] == 'ARG', 'rate_deaths_cases'].mean())
print('Máximo índice de restrição da América do Sul:', df_new.loc[:, 'stringency_index'].max())

# 7.1 - Dispersão
### Usando pandas
df_new['total_cases'].plot();
plt.title('Total de casos - Usando Pandas')
plt.show()
### Usando matplotlib
sns.scatterplot(x=df_new.index, y=df_new['total_cases'], hue=df_new['iso_code'])
plt.title('Total de casos por país - Usando Seaborn')
plt.show();
### Gráfico de linha
#### Dimensoes da figura
sns.lineplot(x=df_new['date'], y=df_new['total_cases'], hue=df_new['iso_code'])
plt.xticks(rotation=90)
plt.title('Total de casos por país - Usando date no eixo x')
## Plot ajustado
plt.show();

# 7.2 - Boxplot (+Violin)
## Boxplot
### Pandas
df_new.loc[:, ['total_deaths']].plot.box()
plt.title('Boxplot usando Pandas')
plt.show();
### Seaborn
sns.boxplot(x=df_new['iso_code'], y=df_new['total_deaths'])
plt.title('Boxplot usando Seaborn')
plt.show();
## Violin
sns.violinplot(x='iso_code', y='stringency_index', data=df_new)
plt.title('Violinplot - Seaborn')
plt.show();
## Matrix
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1,2,1)
sns.violinplot(x='iso_code', y='stringency_index', data=df_new, ax=ax)
plt.title('Violin')
ax = fig.add_subplot(1,2,2)
sns.boxplot(x='iso_code', y='stringency_index', data=df_new, ax=ax)
plt.title('Boxplot')
plt.show();

# 7.3 - Histograma (+Density)
## Usando o Pandas
df_new[['stringency_index']].plot.hist()
plt.title('Histograma - Usando pandas')
plt.show();
## Acumulando variáveis - Usando Pandas
df_new['total_deaths'].plot.hist()
df_new['new_cases'].plot.hist()
plt.title('Histograma com mais variáveis - Usando pandas')
plt.legend()
plt.show();
## Usando Seaborn
plt.figure(figsize=(10, 7))
sns.histplot(data=df_new, x="new_cases", hue="iso_code", kde=False)
plt.show();
## Usando Seaborn - Adicionando densidade
plt.figure(figsize=(10, 7))
sns.histplot(data=df_new, x="new_cases", hue="pop", kde=True)
plt.show();
## Gráfico de densidade - Usando Seaborn
sns.kdeplot(data=df_new, x="new_deaths")
plt.show();
## Gráfico de densidade acumulado - Usando Seaborn
sns.kdeplot(data=df_new, x="new_deaths", hue="iso_code", cumulative=True, common_norm=False, common_grid=True)
plt.show();

# 7.4 - Barras (+Pie)
## Usando Pandas
df_new.loc[:, ['new_cases', 'new_deaths']].max().plot.bar()
plt.title('Gráfico de barras - Usando Pandas')
plt.show();
## Usando Seaborn
df_niver = df_new.loc[df_new['date'] == '2021-02-01', :]
sns.barplot(x="iso_code", y="new_deaths", data=df_niver)
plt.title('Gráfico de barras - Usando Seaborn')
plt.show();
## Gráfico de pizza - Usando Pandas
df_niver.index = df_niver['iso_code']
df_niver.plot.pie(y='new_deaths', figsize=(5, 5))
plt.title('Plot Pie - Usando Pandas')
plt.show();
## Gráfico de pizza com detalhes - Usando pandas
df_niver.index = df_niver['iso_code']
df_niver.plot.pie(y='total_deaths', figsize=(10, 10), autopct='%1.1f%%')
plt.title('Plot Pie com % - Usando Pandas')
plt.show();

# 7.5 - Correlograma (+Heatmap)
## Exibindo correlograma - Pearson
print('Correlação - Pearson')
display(df_new.corr(method='pearson'))
## Exibindo correlograma - Spearman
print('\nCorrelação - Spearman')
display(df_new.corr(method='spearman'))
print('\n\n')
## Heatmap a partir de pearson - Usando Seaborn
sns.heatmap(df_new.corr())
plt.show();
## Heatmap com detalhes - Usando Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df_new.corr(), vmin=-1, vmax=1, annot=True, annot_kws={"size": 12}, cmap="coolwarm")
plt.show();

# 7.6 - Combinados
## pairplot
### Simples
sns.pairplot(df_new.loc[:, ['total_cases', 'total_deaths', 'stringency_index', 'rate_deaths_cases']])
plt.show();
### Adicionando detalhes
sns.pairplot(df_new.loc[:, ['iso_code', 'total_cases', 'total_deaths', 'stringency_index', 'rate_deaths_cases']], hue="iso_code", diag_kind="kde")
plt.show();
## jointplot
### Simples
sns.jointplot(data=df_new, x="new_cases", y="stringency_index") # kind="hex"
plt.show();
### Adicionando detalhes
sns.jointplot(data=df_new, x="new_cases", y="stringency_index", hue="pop", kind="kde")
plt.show();
