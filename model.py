import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_csv(path):

    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, "Data", path + ".csv"))

    return data

def plot_hist(df):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)
    
    plt.figure()
    plt.style.use("ggplot")
    features.hist(bins=50, figsize=(10,5))
    plt.title("Distribución de descriptores Biosurfactantes")
    plt.xlabel("Descriptor")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):

    features = df.drop(['Name', 'Fasta', 'Label'], axis=1)

    corr_matrix = features.corr()
    matrix = np.triu(corr_matrix)
    
    plt.figure()
    plt.title("Matriz de Correlación entre Descriptores")
    sns.set(rc = {'figure.figsize':(15,8)})
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


if __name__ == "__main__":

    train = read_csv("train")
    test = read_csv("test")

    plot_hist(train)

    correlation_analysis(train)

    scatter_plot(train, "Gravy", "Helix")

    pca_analysis(train)

    model = kmeans_model(2, train)

    plot_kmeans(train, model, "InstabilityIndex", "MolWeight")

    test_kmeans(test, model, "InstabilityIndex", "MolWeight")

