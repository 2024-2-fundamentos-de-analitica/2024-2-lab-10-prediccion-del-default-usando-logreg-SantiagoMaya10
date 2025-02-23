
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os


def limpieza(df):
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"]!=0) & (df["MARRIAGE"]!=0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x>4 else x) 
    return df

def crear_pipeline(categorical_features, k_features=10):
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
        ], 
        remainder='passthrough' 
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k_features)),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    
    return pipeline
def optimizar_hiperparametros(pipeline, x_train, y_train):
    param_grid = {
        'selector__k': range(1, 11),
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty':['l1','l2'],
        'classifier__solver':['liblinear'],
        "classifier__max_iter": [100,200],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,           
        param_grid=param_grid,       
        cv=10,                       
        scoring='balanced_accuracy',  
        n_jobs=-1,
        refit=True,
        verbose=3)    
    return grid_search

def calcular_metricas(y_true, y_pred, dataset):
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def calcular_matriz_confusion(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

def save_model(path: str, estimator: GridSearchCV):
    with gzip.open(path, 'wb') as f:
        pickle.dump(estimator, f)

def main():
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_data=limpieza(test_data)
    train_data=limpieza(train_data)
    x_train=train_data.drop('default', axis=1)
    y_train=train_data['default']
    x_test=test_data.drop('default', axis=1)
    y_test=test_data['default']
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()


    pipeline = crear_pipeline(categorical_features)
    grid_search = optimizar_hiperparametros(pipeline, x_train, y_train)
    grid_search.fit(x_train, y_train)
    path2 = "./files/models/"
    save_model(
        os.path.join(path2, 'model.pkl.gz'),
        grid_search,
    )
   

    pred_train = grid_search.predict(x_train)
    pred_test = grid_search.predict(x_test)

    metrics = [
        calcular_metricas(y_train, pred_train, 'train'),
        calcular_metricas(y_test, pred_test, 'test'),
        calcular_matriz_confusion(y_train, pred_train, 'train'),
        calcular_matriz_confusion(y_test, pred_test, 'test')
    ]
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
   

if __name__ == "__main__":
    main()