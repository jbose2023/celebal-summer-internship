import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Load Dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data

# 2. Vectorize Text Data (TF-IDF for KMeans, Count for LDA)
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=10, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)

count_vectorizer = CountVectorizer(max_df=0.5, min_df=10, stop_words='english')
count = count_vectorizer.fit_transform(documents)

# 3. Apply KMeans Clustering
k = 20  # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(tfidf)

print("\nTop terms per cluster (KMeans):")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()
for i in range(k):
    print(f"Cluster {i}: ", end='')
    print(', '.join([terms[ind] for ind in order_centroids[i, :10]]))

# Optional: evaluate KMeans clustering with silhouette score
score = silhouette_score(tfidf, kmeans_labels)
print(f"\nSilhouette Score (KMeans): {score:.3f}")

# 4. Apply LDA for Topic Modeling
lda = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=42)
lda.fit(count)

print("\nTop words per topic (LDA):")
terms = count_vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}: ", end='')
    print(", ".join([terms[i] for i in topic.argsort()[:-11:-1]]))
