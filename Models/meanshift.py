import pandas as pd   

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    dataset = pd.read_csv("./data_sets/candy.csv")
    #print(dataset.head(5))
    
    X=dataset.drop('competitorname',axis=1)
    meanshift=MeanShift().fit(X)
    print(meanshift.labels_)
    print(max(meanshift.labels_)) # muestra el valor maximo de etiquetas del cual podemos deducir el numero de grupos en los cuales clasifico
    print("--"*80)
    print(meanshift.cluster_centers_) 

    dataset['meanshiftl'] =meanshift.labels_
    print("--"*80)
    print(dataset)