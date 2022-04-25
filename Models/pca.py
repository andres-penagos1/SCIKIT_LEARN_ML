import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__" :
    #cargando el data set
    dt_heart = pd.read_csv('./data_sets/heart.csv')
    #separando los features de la prediccion
    dt_features = dt_heart.drop(['target'], axis=1) #borrando la columna de target
    dt_target = dt_heart['target']

    #normalizar los datos 
    dt_features= StandardScaler().fit_transform(dt_features)

    #creando el split de datos de entrenamiento y test
    X_train , X_test ,y_train, y_test = train_test_split(dt_features,dt_target,test_size=0.3,random_state=42)

    print(X_train.shape)
    print(y_train.shape)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    #IncrementalPCA no envia todo los datos de golpe, lo hace en batches
    ipca=IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

   
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE PCA",logistic.score(dt_test,y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train,y_train)
    print("SCORE iPCA",logistic.score(dt_test,y_test))
#con pca se generaron 3 features artificiales de 13 que existian 
#originalmente

 


