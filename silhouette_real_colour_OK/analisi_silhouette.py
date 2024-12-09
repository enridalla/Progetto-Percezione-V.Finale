import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from PIL import Image

# Carica l'immagine e preprocessala
image_path = "./COCO/000000003501.jpg"  # Inserisci il percorso dell'immagine
image = Image.open(image_path).convert("RGB")
image = image.resize((100, 100))  # Ridimensiona per ridurre i calcoli (opzionale)

# Converti l'immagine in un array bidimensionale
X = np.array(image).reshape(-1, 3)  # Ogni pixel è un punto con 3 caratteristiche (R, G, B)

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # Calcola i colori medi per ciascun cluster
    cluster_centers = clusterer.cluster_centers_  # Centroidi RGB dei cluster
    cluster_colors = cluster_centers / 255  # Normalizza i colori medi tra 0 e 1

    # Silhouette analysis
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Usa il colore medio del cluster per colorare la silhouette
        color = cluster_colors[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot: scatter plot of original pixel colors
    # Estrarre i valori originali RGB per i colori
    # Genera i colori per ciascun pixel in base ai cluster
    cluster_pixel_colors = cluster_colors[cluster_labels]

    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=10, lw=0, alpha=0.7, c=cluster_pixel_colors, edgecolor="k"
    )

    ax2.set_title(f"Scatter plot of pixels colored by clusters (n_clusters = {n_clusters})")

    ax2.set_xlabel("Red channel intensity")
    ax2.set_ylabel("Green channel intensity")

    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on image data with n_clusters = {n_clusters}",
        fontsize=14,
        fontweight="bold",
    )

plt.show()
