import pandas as pd
from sklearn .model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    dataset = pd.read_csv('./data_sets/felicidad.csv')

    X=dataset.drop(['country','rank','score'],axis=1)
    y=dataset['score']
    ## definir el regresor a utilixar sin ningun parametro 
    ## pues este es el objetivo de usar randomized search
    reg=RandomForestRegressor()
   
    #definir la grilla de parametros que va a utilizar el optimizador

    parametros= {
        'n_estimators':range(4,16), #define cuantos arboles generaran el bosque aleatorio
        'criterion':['squared_error','absolute_error'],  #medida de calidad de los splits del arbol
        'max_depth':range(2,11)     #profundidad del arbol
    }
    
    #aqui le paso los parametros al optimizador : el modelo a optimizar,la grilla de paramertros,
    #n_iter=10 que tomara 10 diferentes configuraciones  de parametros
    #cv=3 numero de plieges de los datos (cross validation)
    #scoring medir que tan bueno fue el modelo
    rand_est= RandomizedSearchCV(reg,parametros,n_iter=10,cv=3,scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]])) #prediccion del score de felicidad para el primer pais del dataset
    
