from matplotlib.pyplot import axis
import pandas as pd
from sklearn import datasets

from sklearn.cluster import MiniBatchKMeans, k_means

if __name__ =='__main__':
    dataset= pd.read_csv('./data_sets/candy.csv')
    

X= dataset.drop('competitorname',axis=1)

k_means = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
                        # se esperan 4 grupos de salida
                        #se enviaran de a 8 datos para hacer el modelo
print("Total de centros: ", len(k_means.cluster_centers_))      
print("=="*64)
print (k_means.predict(X))  ##devuelve un arreglo con cada una de las categorias              

#se crea una columna en el daset donde vaya la prediccion
dataset['group']=k_means.predict(X)
print(dataset.head(5))