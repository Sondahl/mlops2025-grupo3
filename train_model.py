import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import joblib
import os


def train_models(mydf: pd.DataFrame):
    currentpath = os.path.dirname(os.path.abspath(__file__))
    
    lst_n_estimators = [2, 3, 5, 8, 13]
    lst_max_depth = [3, 5, 8, None]
    lst_min_samples_leaf = [8, 5, 3]

    # Divide os dados em X e y, e posteriormente 80% em treino e 20% em teste
    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    X = mydf[independentcols]
    y = mydf['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria o Classifier (modelo 1)
    acuracia=0
    for n_estimators in lst_n_estimators:
      for max_depth in lst_max_depth:
        for min_samples_leaf in lst_min_samples_leaf:
          independentcols_m1 = independentcols.copy()
          clf = rfc(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
          clf.fit(X=X_train[independentcols_m1], y=y_train)
          clf.independentcols = independentcols_m1
          clf_acuracia = clf.score(X=X_test[independentcols_m1], y=y_test)
          if clf_acuracia > acuracia:
            joblib.dump(clf, f'{currentpath}/models/modelo01.joblib')
            acuracia = clf_acuracia
    print("Modelo 01 (classificador), criado com acurácia de: [{0}]".format(acuracia))
    print("Modelo 01 (classificador) salvo com sucesso.")

    # Cria o Regressor (modelo 2)
    acuracia=0
    for n_estimators in lst_n_estimators:
      for max_depth in lst_max_depth:
        for min_samples_leaf in lst_min_samples_leaf:
          independentcols_m2 = independentcols.copy()
          independentcols_m2.remove('etnia')
          rgs = rfr(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
          rgs.fit(X=X_train[independentcols_m2], y=y_train)
          rgs.independentcols = independentcols_m2
          rgs_acuracia = rgs.score(X=X_test[independentcols_m2], y=y_test)
          if rgs_acuracia > acuracia:
            joblib.dump(rgs, f'{currentpath}/models/modelo02.joblib')
            acuracia = rgs_acuracia
    print("Modelo 02 (Regressor), criado com acurácia de: [{0}]".format(acuracia))
    print("Modelo 02 (regressor) salvo com sucesso.")

if __name__ == "__main__":
    # Carrega os dados
    mydf = pd.read_csv('./datasets/BaseDefault01.csv')
    train_models(mydf=mydf)