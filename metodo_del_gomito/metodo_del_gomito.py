import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

def plot_elbow_method(image_path, max_k=35):
    """
    Plotta il grafico del metodo del gomito per trovare il numero ottimale di cluster.
    :param image_path: Percorso dell'immagine da segmentare.
    :param max_k: Numero massimo di cluster da valutare.
    """
    # Carica l'immagine e prepara i dati
    image = io.imread(image_path)
    pixels = image.reshape((-1, 3)) / 255.0  # Normalizza i pixel
    
    inertias = []
    
    # Calcola l'inertia per ogni valore di k
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        inertias.append(kmeans.inertia_)
    
    # Plot del grafico del gomito
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='--')
    plt.title('Metodo del Gomito per Determinare il Numero di Cluster')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()
