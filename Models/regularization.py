import pandas as pd
import sklearn
from sklearn import datasets

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data_sets/felicidad.csv')

    X = dataset[['gdp','family','lifexp','freedom','corruption','generosity','dystopia']]
    y = dataset[['score']]

    X_train, X_test ,y_train,y_test = train_test_split(X,y,test_size=0.25)

    modellinear = LinearRegression().fit(X_train,y_train)
    y_predict_linear= modellinear.predict(X_test)

    modelLasso = Lasso(alpha=0.02).fit(X_train,y_train) # el valor alpha es el respondable de castigar los features, si el valor es mas grande castiga mas los features
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train,y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    linear_loss= mean_squared_error(y_test,y_predict_linear)
    print("Linear Loss", linear_loss)

    lasso_loss= mean_squared_error(y_test,y_predict_lasso)
    print("Lasso Loss", lasso_loss)

    Ridge_loss= mean_squared_error(y_test,y_predict_ridge)
    print("ridge Loss", Ridge_loss)
    
    print("=" * 70) ## esto simplemente es para crear una linea divisoria en la terminal (Asunto estetico)

    print("Coef LASSO")
    print(modelLasso.coef_ )

    print("Coef Ridge")
    print(modelRidge.coef_ )






