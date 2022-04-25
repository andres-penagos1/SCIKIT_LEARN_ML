##la idea de este codigo es  ver comom mejora el claificador usando el metodo bagging

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dt_heart = pd.read_csv('./data_sets/heart.csv')
    print(dt_heart['target'].describe())

    X=dt_heart.drop(['target'],axis=1)
    y=dt_heart['target']

    X_train,X_test ,y_train,y_test = train_test_split(X,y,test_size=0.35)

    #vamos a usar el clasificador KNeighbors  
    knn_class =KNeighborsClassifier().fit(X_train,y_train)
    knn_pred =knn_class.predict(X_test)
    print("="*80)
    print (accuracy_score(knn_pred,y_test)) ## comparamos la prediccion con los targets de prueba

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(),n_estimators=50).fit(X_train,y_train)
                                  ## este metodo pide dos parametros el tipo de estimador y el numero de estimador
    bag_pred = bag_class.predict(X_test)
    print("="*80)
    print(accuracy_score(bag_pred,y_test))# comparo la prediccion de bagg con los target de test



