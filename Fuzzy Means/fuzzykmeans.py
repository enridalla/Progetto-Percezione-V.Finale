import numpy as np
import matplotlib.pyplot as plt
from skimage import io

class FuzzyKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tolerance=1e-4, m=2.0):
        self.n_clusters = n_clusters         # Numero di cluster
        self.max_iter = max_iter             # Numero massimo di iterazioni
        self.tolerance = tolerance           # Tolleranza per la convergenza
        self.m = m                           # Parametro di fuzzyficazione (m > 1)

    def fit(self, X):
        """
        Esegui l'algoritmo fuzzy K-means per segmentare i dati (o i pixel dell'immagine)
        :param X: Matrice dei dati (o pixel) da segmentare
        :return: Segmentazione dei dati (o dell'immagine) e i centri dei cluster
        """
        # Passo 1: Inizializzazione dei centri dei cluster
        print("Inizializzazione dei centri dei cluster")
        n_samples = X.shape[0]
        centers = np.random.rand(self.n_clusters, X.shape[1])  # Centri iniziali casuali
        U = np.random.rand(n_samples, self.n_clusters)  # Membresia casuale

        # Normalizzazione delle memberships
        U = U / np.sum(U, axis=1, keepdims=True)

        for i in range(self.max_iter):
            # Passo 2: Calcolo delle distanze ponderate
            U_old = U.copy()
            U = self.__update_membership(X, centers)

            # Passo 3: Calcolo dei nuovi centri
            centers = self.__compute_centers(X, U)

            # Verifica la convergenza
            if np.linalg.norm(U - U_old) < self.tolerance:
                print("Convergenza raggiunta.")
                break

        return U, centers

    def __update_membership(self, X, centers):
        """
        Aggiorna le membership in base alla distanza e al parametro di fuzzyficazione
        """
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)  # Distanze Euclidee
        # Calcola la membership utilizzando la formula di fuzzy c-means
        U = 1.0 / (distances ** (2 / (self.m - 1)))  # Potenza basata su m
        U = U / np.sum(U, axis=1, keepdims=True)  # Normalizza le membership
        return U

    def __compute_centers(self, X, U):
        """
        Calcola i centri dei cluster pesati dalle memberships
        """
        num = np.dot(U.T ** self.m, X)  # Pesi delle membership
        denom = np.sum(U.T ** self.m, axis=1)[:, np.newaxis]
        return num / denom

# Funzione per segmentare l'immagine
def fuzzy_kmeans_segment_image(image_path, n_clusters=3):
    image = io.imread(image_path)
    pixels = image.reshape((-1, 3)) / 255.0  # Normalizza i pixel dell'immagine

    fuzzy_kmeans = FuzzyKMeans(n_clusters=n_clusters)
    memberships, centers = fuzzy_kmeans.fit(pixels)

    # Assegna i colori medi ai pixel
    labels = np.argmax(memberships, axis=1)  # Assegna il cluster con membership più alta
    segmented_image = centers[labels]
    segmented_image = (segmented_image * 255).astype(np.uint8)  # Riporta i colori a 0-255
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image, image, memberships, centers, pixels

def display_segmented_image(segmented_image, original_image, memberships, centers, pixels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Immagine Originale')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_image)
    axes[1].set_title('Immagine Segmentata')
    axes[1].axis('off')
    
    plt.show()

    # Visualizzazione dei cluster
    plt.figure(figsize=(8, 6))

    # Mostra i pixel con i loro colori reali
    plt.scatter(pixels[:, 0], pixels[:, 1], c=pixels, marker='o', s=10)
    
    # Aggiungi croci ai centri dei cluster senza etichetta
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], c='red', marker='x', s=100, label='')

    plt.title("Grafico dei Pixel con Centri dei Cluster")
    plt.xlabel('Canale Rosso (R)')
    plt.ylabel('Canale Verde (G)')
    plt.show()

# Esegui il codice per segmentare l'immagine
#image_path = './COCO/cell_00035_label.tiff'
image_path = './COCO/000000003501.jpg'
#image_path = './COCO/pepper2.png'  # Inserisci il percorso dell'immagine
segmented_image, original_image, memberships, centers, pixels = fuzzy_kmeans_segment_image(image_path, n_clusters=3)
display_segmented_image(segmented_image, original_image, memberships, centers, pixels)

