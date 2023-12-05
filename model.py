import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def read_csv(path):

    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, "Data", path + ".csv"))

    return data

def corregir_tension_interfacial(data_frame):
    
    filtered_data = data_frame[~data_frame['Name'].isin(['P0A915', 'P77747', 'P76773', 'P76045', 'P0AA16', 'P09169', 'P06996', 'Mutant6', 'Mutant12'])]

    
    new_tension_interfacial = (filtered_data['I'] + filtered_data['ChargeAtPH7'] + filtered_data['Turn'] + filtered_data['AliphaticIndex']) / 4

    
    data_frame.loc[~data_frame['Name'].isin(['P0A915', 'P77747', 'P76773', 'P76045', 'P0AA16', 'P09169', 'P06996', 'Mutant6', 'Mutant12']), 'TensionInterfacial'] = new_tension_interfacial

    return data_frame

def plot_hist(df, output_dir='histograms'):

    # Crear un directorio para almacenar las imágenes si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Excluir columnas no numéricas
    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)
    numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    for column in numerical_columns:
        plt.figure()
        plt.style.use("ggplot")
        df[column].hist(bins=50)
        plt.title(f"Distribución de {column}")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        # Guardar la imagen en el directorio
        output_path = os.path.join(output_dir, f"{column}_histogram.png")
        plt.savefig(output_path)
        plt.close()

def correlation_analysis(df):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    corr_matrix = features.corr()
    matrix = np.triu(corr_matrix)
    
    print(corr_matrix['TensionInterfacial'])
    
    plt.figure()
    plt.title("Matriz de Correlación entre Descriptores")
    sns.set(rc = {'figure.figsize':(30,16)})
    sns.heatmap(corr_matrix, annot = True, mask=matrix)
    plt.show()

def scatter_plot(df, x_var, y_var):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    x = features[x_var]
    y = features[y_var]

    plt.figure()
    plt.style.use("ggplot")
    plt.scatter(x, y, c ="blue")
    plt.title("Scatter Plot entre " + y_var + " y " + x_var)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()

def pca_analysis(df):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    features['PCA_1'] = X_pca[:, 0]
    features['PCA_2'] = X_pca[:, 1]

    pca_components = pca.components_

    index_max1 = np.argmax(pca_components[0])
    index_max2 = np.argmax(pca_components[1])

    names_features = features.columns

    print(f"La característica más relevante en la PCA1 es {names_features[index_max1]}")
    print(f"La característica más relevante en la PCA2 es {names_features[index_max2]}")

    plt.figure()
    plt.style.use("ggplot")
    plt.scatter(features['PCA_1'], features['PCA_2'])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Análisis de Componentes Principales (PCA) Biosurfactantes')
    plt.show()

def kmeans_model(k, df):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    random_seed = 42
    kmeans = KMeans(k, random_state = random_seed)

    kmeans.fit(features)

    return kmeans

def plot_kmeans(df, model, feature1, feature2):

    df["clustered_label"] = model.labels_
    colors = ["mediumseagreen", "lightcoral"]
    
    plt.figure()
    plt.style.use("ggplot")
    for clustered_label, color in zip(range(len(colors)), colors):
        clustered_points = df[df['clustered_label'] == clustered_label]
        plt.scatter(clustered_points[feature1], clustered_points[feature2], c=color, label=f'Grupo {clustered_label}')

    plt.title('Asignaciones de datos entrenamiento con K-Means')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    plt.show()

def test_kmeans(df, model, feature1, feature2):

    labels = df["Label"]
    colors = ["mediumseagreen", "lightcoral"]
    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    df["clustered_label"] = model.predict(features)

    plt.figure()
    plt.style.use("ggplot")
    for clustered_label, color in zip(range(len(colors)), colors):
        clustered_points = df[df['clustered_label'] == clustered_label]
        plt.scatter(clustered_points[feature1], clustered_points[feature2], c=color, label=f'Grupo {clustered_label}')

    plt.title('Asignaciones de datos con K-Means')
    plt.legend()
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.show()

    f_score = f1_score(labels, df["clustered_label"], average='weighted')
    print(f"F-score en el conjunto de prueba: {round(f_score * 100, 2)}%")

def train_evaluate_svm(train_features, train_labels, test_features, test_labels, param_grid=None):
    """
    Entrenar y evaluar un modelo SVM.

    Parameters:
    - train_features: DataFrame, características de entrenamiento.
    - train_labels: Series, etiquetas de entrenamiento.
    - test_features: DataFrame, características de prueba.
    - test_labels: Series, etiquetas de prueba.
    - param_grid: dict, diccionario de hiperparámetros para GridSearchCV. (Opcional)

    Returns:
    - clf: modelo SVM entrenado.
    """
    svm = SVC()
    
    if param_grid:
        grid_search = GridSearchCV(svm, param_grid, cv=5)
        clf = grid_search.fit(train_features, train_labels)
    else:
        clf = svm.fit(train_features, train_labels)

    # Evaluación en el conjunto de prueba
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy en el conjunto de prueba: {round(accuracy * 100, 2)}%")
    print("Reporte de clasificación:")
    print(classification_report(test_labels, predictions))

    return clf

def train_evaluate_random_forest(train_features, train_labels, test_features, test_labels, param_grid=None):
    """
    Entrenar y evaluar un modelo Random Forest.

    Parameters:
    - train_features: DataFrame, características de entrenamiento.
    - train_labels: Series, etiquetas de entrenamiento.
    - test_features: DataFrame, características de prueba.
    - test_labels: Series, etiquetas de prueba.
    - param_grid: dict, diccionario de hiperparámetros para GridSearchCV. (Opcional)

    Returns:
    - clf: modelo Random Forest entrenado.
    """
    rf = RandomForestClassifier()
    
    if param_grid:
        grid_search = GridSearchCV(rf, param_grid, cv=5)
        clf = grid_search.fit(train_features, train_labels)
    else:
        clf = rf.fit(train_features, train_labels)

    # Evaluación en el conjunto de prueba
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy en el conjunto de prueba: {round(accuracy * 100, 2)}%")
    print("Reporte de clasificación:")
    print(classification_report(test_labels, predictions))

    return clf


if __name__ == "__main__":

    raw_train = read_csv("train")
    raw_test = read_csv("test")

    data = raw_train[raw_train['TensionInterfacial'] != 0]
    correlation_analysis(data)

    train = corregir_tension_interfacial(raw_train)
    test = corregir_tension_interfacial(raw_test)

    plot_hist(train)

    scatter_plot(train, "MolWeight", "Helix")

    pca_analysis(train)

    model = kmeans_model(2, train)

    plot_kmeans(train, model, "MolWeight", "Helix")

    test_kmeans(test, model, "MolWeight", "Helix")

     # Ejemplo de entrenamiento y evaluación de SVM
    svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm_model = train_evaluate_svm(train.drop(['Name', 'Fasta', 'Label'], axis=1), train['Label'],
                                    test.drop(['Name', 'Fasta', 'Label'], axis=1), test['Label'],
                                    param_grid=svm_param_grid)

    # Ejemplo de entrenamiento y evaluación de Random Forest
    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    rf_model = train_evaluate_random_forest(train.drop(['Name', 'Fasta', 'Label'], axis=1), train['Label'],
                                            test.drop(['Name', 'Fasta', 'Label'], axis=1), test['Label'],
                                            param_grid=rf_param_grid)
