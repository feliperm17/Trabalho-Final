import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#from sklearn
from sklearn.metrics import classification_report, accuracy_score

# Carregar os dados do arquivo CSV
data = pd.read_csv('./features.csv')
data_test = pd.read_csv('./featuresTeste.csv')
df = pd.DataFrame(data)
df_test = pd.DataFrame(data_test)
# Separar as características (features) e os rótulos (labels)
x_train = data.iloc[:, 1:]  # Todas as colunas exceto a primeira
y_train = data.iloc[:, 0]   # Primeira coluna (todas as linhas)
print (x_train.head())
print (y_train.head())
#X_train = df.drop(0, axis=1)
#y_train = df[0]

x_test = data_test.iloc[:, 1:]
y_test = data_test.iloc[:, 0] 
#X_test = data_test.drop(0, axis=1)
#y_test = data_test[0]



# Criar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn.fit(x_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn.predict(x_test)

# Avaliar o modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
