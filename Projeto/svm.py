import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.fixes import delayed
from joblib import Parallel, parallel_backend

# Carregar os dados do arquivo CSV
data = pd.read_csv('./featureslpq.csv')
data_test = pd.read_csv('./featureslpqtest.csv')

# Separar as características (features) e os rótulos (labels)
x_train = data.iloc[:, 1:]  # Todas as colunas exceto a primeira (features)
y_train = data.iloc[:, 0]   # Primeira coluna (labels)

x_test = data_test.iloc[:, 1:]  # Features de teste
y_test = data_test.iloc[:, 0]   # Labels de teste

# Criar Modelo SVM com Kernel RBF
svm = SVC(kernel='rbf', probability=True)

# Parâmetros para Grid Search
params = {
    "svm__C": [0.1, 1, 10, 100, 1000],
    "svm__gamma": [2e-5, 2e-3, 2e-1, "auto", "scale"]
}

# Pipeline para escalonamento e modelo
pipe = Pipeline([
    ("scaler", StandardScaler()), 
    ("svm", svm)
])

# Customização do GridSearchCV com tqdm para barra de progresso
class TqdmParallel(Parallel):
    def __init__(self, total=None, **kwargs):
        super().__init__(**kwargs)
        self._total = total

    def __call__(self, *args, **kwargs):
        with tqdm(total=self._total, desc="GridSearch Progress") as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.update(1)

# Calcula o número total de combinações de parâmetros para tqdm
total_comb = len(params['svm__C']) * len(params['svm__gamma'])

# Otimização de hiperparâmetros com GridSearchCV e barra de progresso
modelo = GridSearchCV(pipe, params, n_jobs=-1, verbose=0)

with parallel_backend('threading', n_jobs=-1):
    with TqdmParallel(total=total_comb) as parallel:
        modelo.fit(x_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.4f}')

# Exibir a matriz de confusão
cm_display = ConfusionMatrixDisplay.from_estimator(modelo, x_test, y_test)
cm_display.plot()
plt.show()

# Relatório de classificação
print(classification_report(y_test, y_pred))
