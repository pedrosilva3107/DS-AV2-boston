#Abrir os dados
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
from pickle import dump

dados = pd.read_csv('HousingData.csv', sep=',')

#A base tem valores vazios (NA), o KMeans não aceita
#numéricos: preencher com a média da coluna
#categóricos: preencher com o valor que mais aparece (moda)

#1 Separar os dados numéricos dos dados categóricos
#CHAS: 1 = perto do rio Charles, 0 = longe do rio (é categoria)
#RAD: índice de acesso às rodovias (é um código, não uma quantidade)
dados_cat_rotulos = ['CHAS', 'RAD']
dados_num = dados.drop(columns=dados_cat_rotulos)
dados_cat = dados[dados_cat_rotulos]

dados_num = dados_num.fillna(dados_num.mean())
dados_cat = dados_cat.fillna(dados_cat.mode().iloc[0]).astype(int)

#2 NORMALIZAR OS DADOS
#2.1 Instanciar o minmax para os numéricos
normalizador = MinMaxScaler()
#2.2 Treinar o modelo normalizador com os dados numéricos
modelo_normalizador = normalizador.fit(dados_num)
#2.3 Salvar o modelo normalizador para uso posterior
dump(modelo_normalizador, open('normalizador_boston.model', 'wb'))
#2.4 Normalizar os dados numéricos
dados_num_normalizados = modelo_normalizador.transform(dados_num)
#2.5 Converter para DataFrame novamente
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

#2.6 Normalizar os categóricos com o get_dummies
dados_cat_normalizados = pd.get_dummies(dados_cat, columns=dados_cat_rotulos, prefix=dados_cat_rotulos, dtype=int)

#2.7 Juntar os dados normalizados em um dataframe único
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)

#3 Descobrir o número ideal de clusters
#HIPERPARAMETRIZAÇÃO
distorcoes = []
K = range(1, 101)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1)/dados_normalizados.shape[0]
            )
        )

#Plotar o gráfico das distorções (só para visualizar)
# from matplotlib import pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(K, distorcoes)
# ax.set(xlabel='n Clusters', ylabel='Distorcoes')
# ax.grid()
# plt.show()

#MÉTODO DO COTOVELO (ELBOW)
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print('Numero otimo de clusters:', n_clusters_otimo)

#4 Treinamento definitivo
boston_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42).fit(dados_normalizados)

#Salvar o modelo para uso posterior
dump(
    boston_cluster,
    open('boston_cluster.model', 'wb')
)
