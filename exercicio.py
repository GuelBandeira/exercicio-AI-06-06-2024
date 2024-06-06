
#IMPORTAÇÃO DAS BIBLIOTECAS
#ALUNOS: João victor silveira domingues - 20210687; José Miguel Bandeira de Novaes - 202120609

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

#setup
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=3000, centers=4,
                       cluster_std=0.90, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

#DEFINIÇÃO DO PRIMEIRO KMEANS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


#PRIMEIRO RESULTADO COM 5 CLUSTERS 
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


#NOVA INSTÂNCIA DE KMEANS
new_kmeans = KMeans(n_clusters=8)
new_kmeans.fit(X)
y_new_kmeans = new_kmeans.predict(X)

#RESULTADO FINAL DO SEGUNDO
plt.scatter(X[:, 0], X[:, 1], c=y_new_kmeans, s=50, cmap='viridis')
centers = new_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
