import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

# Numero di cluster da utilizzare
N_CLUSTERS = 10  # Puoi cambiare questo valore per modificare il numero di cluster ovunque

class KMeans:
    def __init__(self, n_clusters=N_CLUSTERS, max_iter=100, tolerance=0.01):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def fit(self, X):
        """
        Esegui l'algoritmo K-means per segmentare i dati (o i pixel dell'immagine)
        :param X: Matrice dei dati (o pixel) da segmentare
        :return: Segmentazione dei dati (o dell'immagine) e i centri dei cluster
        """
        # Passo 1: Inizializzazione dei centri dei cluster
        print("Passo 1: Inizializzazione dei centri dei cluster")
        cluster_centers = self.__initialize_means(X)
        print(f"Inizializzazione: {cluster_centers}")
        
        for i in range(self.max_iter):
            # Passo 2: Assegnazione dei punti ai cluster più vicini
            print(f"Passo 2: Iterazione {i+1} - Assegnazione dei punti ai cluster")
            labels = self.__assign_labels(X, cluster_centers)
            print(f"Etichette assegnate: {labels}")
            
            # Passo 3: Ricalcolo dei centri dei cluster
            print("Passo 3: Ricalcolo dei centri dei cluster")
            new_cluster_centers = self.__compute_centers(X, labels)
            print(f"Nuovi centri: {new_cluster_centers}")
            
            # Verifica se i centri sono cambiati abbastanza
            if self.__convergence_check(cluster_centers, new_cluster_centers):
                print("Algoritmo convergente, i centri non cambiano più.")
                break
            cluster_centers = new_cluster_centers
        
        return labels, cluster_centers
    
    def __initialize_means(self, X):
        """Inizializza i centri dei cluster con il metodo Forgy (seleziona punti casuali come centri)"""
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def __assign_labels(self, X, cluster_centers):
        """Assegna ad ogni punto l'etichetta del cluster più vicino"""
        distances = np.linalg.norm(X[:, np.newaxis] - cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def __compute_centers(self, X, labels):
        """Ricalcola i centri dei cluster come la media dei punti assegnati a ciascun cluster"""
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers
    
    def __convergence_check(self, old_centers, new_centers):
        """Verifica se i centri sono cambiati abbastanza (entro la tolleranza)"""
        return np.all(np.abs(old_centers - new_centers) < self.tolerance)

# Funzione per segmentare l'immagine
def segment_image(image_path, n_clusters=N_CLUSTERS):
    image = io.imread(image_path)
    pixels = image.reshape((-1, 3)) / 255.0  # Normalizza i pixel dell'immagine
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels, cluster_centers = kmeans.fit(pixels)
    
    # Assegna i colori medi ai pixel
    segmented_image = cluster_centers[labels]
    segmented_image = (segmented_image * 255).astype(np.uint8)  # Riporta i colori a 0-255
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image, image, labels, cluster_centers, pixels

# Funzione per visualizzare l'immagine segmentata e il grafico dei cluster
def display_segmented_image(segmented_image, original_image, labels, cluster_centers, pixels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Immagine Originale')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_image)
    axes[1].set_title('Immagine Segmentata')
    axes[1].axis('off')
    
    plt.show()

    # Visualizzazione dei cluster con i centroidi
    plt.figure(figsize=(8, 6))

    # Prendiamo i canali R e G per visualizzare in 2D (per esempio, possiamo ignorare il canale B)
    plt.scatter(pixels[:, 0], pixels[:, 1], c=labels, cmap='viridis', marker='o', label="Pixel", s=10)

    # Visualizza i centroidi
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100, label="Centroidi")
    
    plt.title("Grafico dei Cluster con Centroidi")
    plt.xlabel('Canale Rosso (R)')
    plt.ylabel('Canale Verde (G)')
    plt.legend()
    plt.show()

# Funzione per visualizzare SOLO il grafico dei cluster con i colori veri dei cluster
def display_segmented_image_real_color(segmented_image, original_image, labels, cluster_centers, pixels):
    
    # Assegna i colori medi (dei centroidi) ai punti
    cluster_colors = cluster_centers[labels]

    # Visualizzazione dei cluster con i centroidi
    plt.figure(figsize=(8, 6))

    # Prendiamo i canali R e G per visualizzare in 2D (per esempio, possiamo ignorare il canale B)
    # Visualizza i cluster con i colori reali
    plt.scatter(pixels[:, 0], pixels[:, 1], c=cluster_colors, marker='o', label="Pixel", s=10)
    
   # Visualizza i centroidi (puoi scegliere un colore unico per evidenziarli)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100, label="Centroidi")
    
    plt.title("Grafico dei Cluster con Centroidi")
    plt.xlabel('Canale Rosso (R)')
    plt.ylabel('Canale Verde (G)')
    plt.legend()
    plt.show()

# Funzione per calcolare i valori medi/mediani dei cluster
def calculate_cluster_statistics(pixels, labels, n_clusters=N_CLUSTERS):
    """
    Calcola i valori medi e mediani per ciascun cluster.
    :param pixels: Array dei pixel normalizzati (Nx3).
    :param labels: Array delle etichette dei cluster assegnate ai pixel.
    :param n_clusters: Numero di cluster.
    :return: Dizionario con valori medi e mediani per ciascun cluster.
    """
    cluster_stats = {}
    
    for cluster_id in range(n_clusters):
        # Filtra i pixel appartenenti al cluster corrente
        cluster_pixels = pixels[labels == cluster_id]
        
        # Calcola media e mediana
        mean_values = cluster_pixels.mean(axis=0) if len(cluster_pixels) > 0 else [0, 0, 0]
        median_values = np.median(cluster_pixels, axis=0) if len(cluster_pixels) > 0 else [0, 0, 0]
        
        # Salva i risultati
        cluster_stats[cluster_id] = {
            "mean": mean_values,
            "median": median_values
        }
    
    return cluster_stats


# Esegui il codice per segmentare l'immagine
#image_path = './COCO/cell_00035_label.tiff'
#image_path = './COCO/000000003501.jpg'
image_path = './COCO/pepper2.png'  # Inserisci il percorso dell'immagine

segmented_image, original_image, labels, cluster_centers, pixels = segment_image(image_path)
display_segmented_image(segmented_image, original_image, labels, cluster_centers, pixels)
display_segmented_image_real_color(segmented_image, original_image, labels, cluster_centers, pixels)

# Calcola valori medi/mediani dei cluster
cluster_stats = calculate_cluster_statistics(pixels, labels)

# Visualizza i risultati
for cluster_id, stats in cluster_stats.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Media: {stats['mean']}")
    print(f"  Mediana: {stats['median']}")