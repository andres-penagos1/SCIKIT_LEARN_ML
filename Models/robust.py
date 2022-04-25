import pandas as pd
from sklearn import datasets

from sklearn.linear_model import (
    RANSACRegressor,HuberRegressor
)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data_sets/felicidad_corrupt.csv')
    
    X = dataset.drop(['country','score'],axis=1)
    y = dataset[['score']]

    X_train, X_test ,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    estimadores = {
        'SVR':SVR(gamma='auto',C= 1.0,epsilon=0.1),
        'RANSAC':RANSACRegressor(),
        'HUBER':HuberRegressor(epsilon=1.35)
    }

    for name, estimadores in estimadores.items():
        estimadores.fit(X_train,y_train)
        predictions = estimadores.predict(X_test)
        print("-"*80)
        print(name)
        print("MSE",mean_squared_error(y_test,predictions))
        
