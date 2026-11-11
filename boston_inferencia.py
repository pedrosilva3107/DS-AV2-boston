#Receber uma nova residência e determinar a qual cluster ela pertence
import pandas as pd
from pickle import load

#CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
nova_residencia = [[0.25, 0, 8.14, 0, 0.538, 6.1, 85.0, 4.2, 4, 307, 21.0, 390.0, 14.5, 19.5]]
nova_residencia = pd.DataFrame(nova_residencia, columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

#carregar o normalizador e o modelo
normalizador = load(open('normalizador_boston.model', 'rb'))
boston_cluster = load(open('boston_cluster.model', 'rb'))

#separar numéricos e categóricos
dados_cat_rotulos = ['CHAS', 'RAD']
nova_num = nova_residencia.drop(columns=dados_cat_rotulos)
nova_cat = nova_residencia[dados_cat_rotulos]

#normalizar os numéricos com o normalizador salvo (só transform)
nova_num = normalizador.transform(nova_num)
nova_num = pd.DataFrame(nova_num, columns=normalizador.feature_names_in_)

#normalizar os categóricos com o get_dummies
nova_cat = pd.get_dummies(nova_cat, columns=dados_cat_rotulos, prefix=dados_cat_rotulos, dtype=int)

#juntar e deixar com as mesmas colunas do treino (as categorias que faltam ficam 0)
nova_residencia_normalizada = nova_num.join(nova_cat)
nova_residencia_normalizada = nova_residencia_normalizada.reindex(columns=boston_cluster.feature_names_in_, fill_value=0)

#inferir o cluster da nova residência
cluster_da_residencia = boston_cluster.predict(nova_residencia_normalizada)
print('Cluster da nova residencia:', cluster_da_residencia[0])

#DESNORMALIZAR O CENTROIDE
centroide = pd.DataFrame(boston_cluster.cluster_centers_[cluster_da_residencia], columns=boston_cluster.feature_names_in_)

#numéricos: inverse_transform
centroide_num = normalizador.inverse_transform(centroide[normalizador.feature_names_in_])
centroide_num = pd.DataFrame(centroide_num, columns=normalizador.feature_names_in_)

#categóricos: no centroide as colunas do get_dummies ficam com valores entre 0 e 1
#a categoria do centroide é a coluna com maior valor, ela vira 1 e as outras 0
centroide_chas = centroide.filter(like='CHAS_')
centroide_chas = (centroide_chas == centroide_chas.max(axis=1)[0]).astype(int)
centroide_rad = centroide.filter(like='RAD_')
centroide_rad = (centroide_rad == centroide_rad.max(axis=1)[0]).astype(int)

centroide_cat = pd.from_dummies(centroide_chas, sep='_').join(pd.from_dummies(centroide_rad, sep='_'))

centroide_natural = centroide_num.join(centroide_cat)
centroide_natural = centroide_natural[nova_residencia.columns]
print(centroide_natural.T)
