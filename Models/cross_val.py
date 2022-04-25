##Como implementar cross validation y evaluar su rendimiento

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score,KFold   
)


if __name__ == "__main__":
    dataset = pd.read_csv('./data_sets/felicidad.csv')
    X= dataset.drop(['country','score'],axis=1) ##quitamos estas dos columnas categoricas
    y = dataset['score']

    model = DecisionTreeRegressor()
    score = cross_val_score(model,X,y,cv=3,scoring='neg_mean_squared_error')#cv=3 Me permite elejir el numero de particiones del kflod
    
    print(np.abs(np.mean(score)))
    ##Hasta aqui la manera mas facil de implementar un crossvalidation ver un score
    
    kf = KFold(n_splits=3,shuffle=True,random_state=42)#shuffle dice si deben organizar aletoriamente los datos
    for train , test in kf.split(dataset):
        ##para cada split que aplica la funcion kf a los datos , 
        ##imprimiremos como ha partido los datos en cada split
        print(train)
        print(test)



