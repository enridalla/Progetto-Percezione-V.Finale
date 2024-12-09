import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

class KMeansCustom:
    def __init__(self, n_clusters=30, max_iter=100, tolerance=0.01, init_method="k-means++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_method = init_method
    
    def fit(self, X):
        """
        Esegui l'algoritmo K-means per segmentare i dati (o i pixel dell'immagine)
        :param X: Matrice dei dati (o pixel) da segmentare
        :return: Segmentazione dei dati (o dell'immagine) e i centri dei cluster
        """
        print("Inizializzazione dei centri dei cluster...")
        if self.init_method == "k-means++":
            cluster_centers = self.__initialize_means_kmeans_plus_plus(X)
        else:
            cluster_centers = self.__initialize_means_random(X)
        
        for i in range(self.max_iter):
            # Passo 2: Assegnazione dei punti ai cluster più vicini
            labels = self.__assign_labels(X, cluster_centers)
            
            # Passo 3: Ricalcolo dei centri dei cluster
            new_cluster_centers = self.__compute_centers(X, labels)
            
            # Verifica se i centri sono cambiati abbastanza
            if self.__convergence_check(cluster_centers, new_cluster_centers):
                print(f"Convergenza raggiunta in {i+1} iterazioni.")
                break
            cluster_centers = new_cluster_centers
        
        return labels, cluster_centers
    
    def __initialize_means_random(self, X):
        """Inizializza i centri dei cluster selezionando punti casuali (Forgy)"""
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def __initialize_means_kmeans_plus_plus(self, X):
        """Inizializza i centri dei cluster utilizzando il metodo K-Means++"""
        n_samples = X.shape[0]
        centers = []
        
        # Scegli il primo centro casualmente
        first_index = np.random.randint(0, n_samples)
        centers.append(X[first_index])
        
        # Seleziona gli altri centroidi
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - center, axis=1)**2 for center in centers], axis=0)
            probabilities = distances / np.sum(distances)
            next_index = np.random.choice(n_samples, p=probabilities)
            centers.append(X[next_index])
        
        return np.array(centers)
    
    def __assign_labels(self, X, cluster_centers):
        """Assegna ad ogni punto l'etichetta del cluster più vicino"""
        distances = np.linalg.norm(X[:, np.newaxis] - cluster_centers, axis=2)
        return np.argmin(distances, axis=1)
    
    def __compute_centers(self, X, labels):
        """Ricalcola i centri dei cluster come la media dei punti assegnati a ciascun cluster"""
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
    
    def __convergence_check(self, old_centers, new_centers):
        """Verifica se i centri sono cambiati abbastanza (entro la tolleranza)"""
        return np.all(np.abs(old_centers - new_centers) < self.tolerance)

# Funzione per segmentare l'immagine
def segment_image(image_path, n_clusters):
    image = io.imread(image_path)
    pixels = image.reshape((-1, 3)) / 255.0  # Normalizza i pixel dell'immagine
    
    kmeans = KMeansCustom(n_clusters=n_clusters, init_method="k-means++")
    labels, cluster_centers = kmeans.fit(pixels)
    
    # Assegna i colori medi ai pixel
    segmented_image = cluster_centers[labels]
    segmented_image = (segmented_image * 255).astype(np.uint8)  # Riporta i colori a 0-255
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image, image, labels, cluster_centers, pixels

# Funzione per visualizzare l'immagine segmentata
def display_segmented_image(segmented_image, original_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    #per le immagini nere inserisci cmap='gray'
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Immagine Originale')
    axes[0].axis('off')
    
    #per le immagini nere inserisci cmap='gray'
    axes[1].imshow(segmented_image, cmap='gray')
    axes[1].set_title('Immagine Segmentata')
    axes[1].axis('off')
    
    plt.show()

# Esegui il codice per segmentare l'immagine
if __name__ == "__main__":

    image_path = './COCO/cell_00035_label.tiff'
    #image_path = './COCO/000000003501.jpg'
    #image_path = './COCO/pepper2.png'  # Inserisci il percorso dell'immagine
    
    # Segmenta l'immagine
    segmented_image, original_image, labels, cluster_centers, pixels = segment_image(image_path, n_clusters=33)
    
    # Visualizza il risultato
    display_segmented_image(segmented_image, original_image)

    # Visualizza il grafico del metodo del gomito
    #plot_elbow_method(image_path, max_k=35)



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


display_segmented_image_real_color(segmented_image, original_image, labels, cluster_centers, pixels)