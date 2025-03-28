import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import sys, math, random, json, sklearn, joblib, uuid, os
from azureml.core import Experiment, Workspace, Dataset, Model
from dotenv import load_dotenv

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
    experiment = Experiment(workspace=workspace, name="Classificador_MLStudio")
    for n_estimators in lst_n_estimators:
      for max_depth in lst_max_depth:
        for min_samples_leaf in lst_min_samples_leaf:
          myrunid = str(uuid.uuid1())
          run = experiment.start_logging(run_id=myrunid,
                                    display_name=f"classificador-{myrunid}",
                                    outputs="modelo",
                                    snapshot_directory="dadostreino")
          independentcols_m1 = independentcols.copy()
          clf = rfc(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
          clf.fit(X=X_train[independentcols_m1], y=y_train)
          clf.independentcols = independentcols_m1
          clf_acuracia = clf.score(X=X_test[independentcols_m1], y=y_test)
          run.log("Tipo", "Classificador")
          run.log("acuracia", clf_acuracia)
          run.log("language", "python")
          run.log("python", sys.version)
          run.log("Versao sklearn", sklearn.__version__)
          run.log("criterion", clf.criterion)
          run.log("n_estimators", clf.n_estimators)
          run.log("min_samples_leaf", clf.min_samples_leaf)
          run.log("max_depth",  str(clf.max_depth))
          run.log("dataset_name",  dataset_name)
          run.log("dataset_version",  dataset_version)
          run.log_list("Inputs", independentcols_m1)
          run.log_list("classes", [int(v) for v in clf.classes_])
          run.complete()
          run.wait_for_completion()
          if clf_acuracia > acuracia:
            joblib.dump(clf, f'{currentpath}/models/modelo01.joblib')
            acuracia = clf_acuracia
    print("Modelo 01 (classificador), criado com acurácia de: [{0}]".format(acuracia))
    print("Modelo 01 (classificador) salvo com sucesso.")

    # Cria o Regressor (modelo 2)
    acuracia=0
    experiment = Experiment(workspace=workspace, name="Regressor_MLStudio")
    for n_estimators in lst_n_estimators:
      for max_depth in lst_max_depth:
        for min_samples_leaf in lst_min_samples_leaf:
          myrunid = str(uuid.uuid1())
          run = experiment.start_logging(run_id=myrunid,
                                    display_name=f"Regressor-{myrunid}",
                                    outputs="modelo",
                                    snapshot_directory="dadostreino")
          independentcols_m2 = independentcols.copy()
          independentcols_m2.remove('etnia')
          rgs = rfr(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
          rgs.fit(X=X_train[independentcols_m2], y=y_train)
          rgs.independentcols = independentcols_m2
          rgs_acuracia = rgs.score(X=X_test[independentcols_m2], y=y_test)
          run.log("Tipo", "Regressor")
          run.log("acuracia", rgs_acuracia)
          run.log("language", "python")
          run.log("python", sys.version)
          run.log("Versao sklearn", sklearn.__version__)
          run.log("criterion", rgs.criterion)
          run.log("n_estimators", rgs.n_estimators)
          run.log("min_samples_leaf", rgs.min_samples_leaf)
          run.log("max_depth",  str(rgs.max_depth))
          run.log("dataset_name",  dataset_name)
          run.log("dataset_version",  dataset_version)
          run.log_list("Inputs", independentcols)
          run.log_list("classes", [int(v) for v in clf.classes_])
          run.complete()
          run.wait_for_completion()
          if rgs_acuracia > acuracia:
            joblib.dump(rgs, f'{currentpath}/models/modelo02.joblib')
            acuracia = rgs_acuracia
    print("Modelo 02 (Regressor), criado com acurácia de: [{0}]".format(acuracia))
    print("Modelo 02 (regressor) salvo com sucesso.")

if __name__ == "__main__":
  
  load_dotenv()
  subscription_id = os.getenv("subscription_id")
  resource_group = os.getenv("resource_group")
  workspace_name = os.getenv("workspace_name")
  workspace = Workspace(subscription_id, resource_group, workspace_name)
  
  dataset_name, dataset_version = 'Risco-De-Credito', None
  try:
    dataset = Dataset.get_by_name(workspace, name=dataset_name)
    dataset_version = dataset.version
    mydf = dataset.to_pandas_dataframe()
  except:
    print("Dataset não encontrado")
     
  # Carrega os dados
  # mydf = pd.read_csv('./datasets/BaseDefault01.csv')
  train_models(mydf=mydf)