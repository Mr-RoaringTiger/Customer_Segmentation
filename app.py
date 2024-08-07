import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle

# Load the saved model, scaler, and one-hot encoder
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('onehotencoder.pkl', 'rb') as encoder_file:
    one_hot_encoder = pickle.load(encoder_file)

# Title of the App
st.title("Customer Segmentation App")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Preprocess the input data
    data_encoded = one_hot_encoder.transform(data[['Gender']])
    data_encoded = pd.DataFrame(data_encoded, columns=one_hot_encoder.get_feature_names_out(['Gender']))
    data_preprocessed = pd.concat([data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], data_encoded], axis=1)

    # Scale the input data
    data_scaled = scaler.transform(data_preprocessed)
    
    # Select the number of clusters
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    
    if st.button("Run Clustering"):
        # Apply KMeans Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, init='k-means++', n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        centroids = kmeans.cluster_centers_
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_scaled)
        
        # t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data_scaled)
        
        # Calculate performance metrics
        silhouette_avg = silhouette_score(data_scaled, labels)
        ch_score = calinski_harabasz_score(data_scaled, labels)
        db_score = davies_bouldin_score(data_scaled, labels)
        
        # Display metrics
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
        st.write(f"Calinski-Harabasz Index: {ch_score:.2f}")
        st.write(f"Davies-Bouldin Index: {db_score:.2f}")
        
        # Plot PCA results
        st.subheader("PCA Clustering Visualization")
        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        scatter_pca = ax_pca.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        ax_pca.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
        ax_pca.set_title('Clusters Visualization with PCA')
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        plt.colorbar(scatter_pca, ax=ax_pca, label='Cluster Label')
        st.pyplot(fig_pca)
        
        # Plot t-SNE results
        st.subheader("t-SNE Clustering Visualization")
        fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
        scatter_tsne = ax_tsne.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        ax_tsne.set_title('Clusters Visualization with t-SNE')
        ax_tsne.set_xlabel('Component 1')
        ax_tsne.set_ylabel('Component 2')
        plt.colorbar(scatter_tsne, ax=ax_tsne, label='Cluster Label')
        st.pyplot(fig_tsne)
