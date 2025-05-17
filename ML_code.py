import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("E_Commerce_Data_Set_4034.csv")
df.head()

df.info()

df.isnull().sum()

df.rename(columns={'Genre': 'Gender'}, inplace=True)
df.head()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

print(f"Number of duplicate rows: {duplicates}")


sns.set(style='whitegrid', palette='pastel')
columns = ['Age', 'Annual_Income (£K)', 'Spending_Score']
for col in columns:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=col, kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col} (Histogram)')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=col, color='lightcoral')
    plt.title(f'{col} (Boxplot)')

    plt.tight_layout()
    plt.show()

df.dropna(inplace = True)

# Select relevant numeric features
features_to_scale = ['Gender', 'Age', 'Annual_Income (£K)', 'Spending_Score']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_to_scale])
# Create a new DataFrame for scaled data
df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)
# Optional: keep original unscaled for plotting
df_plot = df.copy()
print(df.head())
df_scaled['Gender'] = df['Gender'].values
print(df_scaled.head())


# Select numeric features (or whichever features you want to cluster on)
X = df.select_dtypes(include=['float64', 'int64'])  # Or specify columns directly
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Updated features with Gender included
features = df_scaled[['Gender', 'Age', 'Annual_Income (£K)', 'Spending_Score']]

# Elbow Method
sse = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(features)
    sse.append(km.inertia_)

# Plot Elbow
plt.figure(figsize=(6, 4))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

print('       ')
# Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(features)
    silhouette_scores.append(silhouette_score(features, labels))

# Plot Silhouette Scores
plt.figure(figsize=(6, 4))
plt.plot(list(range(2, 11)), silhouette_scores, marker='o')
plt.title('Silhouette Scores for K-Means')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
print('       ')
# Choose optimal k and fit KMeans
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(features)
# KMeans Cluster Scatter Plot (2D projection using top two variables)
plt.figure(figsize=(6, 5))
sns.scatterplot(x='Annual_Income (£K)', y='Spending_Score',
                hue='KMeans_Cluster', data=df, palette='Set2')
plt.title('K-Means Clusters')
plt.show()
print('       ')

# 1. Group by cluster and compute mean of numerical features
cluster_profile = df.groupby('KMeans_Cluster')[['Gender', 'Age', 'Annual_Income (£K)', 'Spending_Score']].mean()
# 2. Add cluster sizes
cluster_profile['Count'] = df['KMeans_Cluster'].value_counts().sort_index()
# 3. Derive gender proportions
# Since 1 = Male, 0 = Female → mean = proportion of males
cluster_profile['% Male'] = (cluster_profile['Gender'] * 100).round(1)
cluster_profile['% Female'] = (100 - cluster_profile['% Male']).round(1)
# Drop raw 'Gender' column to avoid confusion
cluster_profile.drop(columns='Gender', inplace=True)
# 4. Display the profile
print("📋 Cluster Profile Summary:")
display(cluster_profile)
print(" ____________________________________________ ")
# 5. Bar plot for average features per cluster
plt.figure(figsize=(12, 6))
cluster_profile.drop(columns=['Count', '% Male', '% Female']).plot(kind='bar')
plt.title("Average Feature Values per Cluster")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Assuming kmeans is your trained model and df_cleaned is the original preprocessed (non-scaled) data
df['Cluster'] = kmeans.labels_
df_scaled['Cluster'] = kmeans.labels_  # Also attach to the scaled version for radar plots

# Bar plot of number of customers in each cluster
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Cluster', palette='Set2')
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()
print("    ")

# Group by cluster and compute mean
cluster_means = df.groupby('Cluster')[['Age', 'Annual_Income (£K)', 'Spending_Score']].mean()

# Plot
cluster_means.plot(kind='bar', figsize=(10,6), colormap='Set3')
plt.title('Average Feature Values per Cluster')
plt.ylabel('Value')
plt.xticks(rotation=0)
plt.legend(title='Feature')
plt.grid(axis='y')
plt.show()
print("    ")


# Prepare features (include encoded Gender now)
features = df_scaled[['Gender', 'Age', 'Annual_Income (£K)', 'Spending_Score']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(features)
df['DBSCAN_Cluster'] = dbscan_labels
# Plot DBSCAN result
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Annual_Income (£K)', y='Spending_Score',
                hue='DBSCAN_Cluster', palette='tab10')
plt.title('DBSCAN Clusters')
plt.legend()
plt.show()
# Silhouette Score for DBSCAN (ignore noise label -1 if exists)
if len(set(dbscan_labels)) > 1 and -1 in set(dbscan_labels):
    mask = dbscan_labels != -1
    dbscan_score = silhouette_score(features[mask], dbscan_labels[mask])
else:
    dbscan_score = 'N/A'
print(f"Silhouette Score for DBSCAN: {dbscan_score}")


# Dendrogram
linkage_matrix = linkage(features, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
print("   ")
# Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=6)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(features)
# Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Annual_Income (£K)', y='Spending_Score',
                hue='Hierarchical_Cluster', palette='Set3')
plt.title('Hierarchical Clusters')
plt.legend()
plt.show()
# Silhouette Score
hierarchical_score = silhouette_score(features, df['Hierarchical_Cluster'])
print(f"Silhouette Score for Hierarchical: {hierarchical_score:.3f}")
spectral = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
spectral_labels = spectral.fit_predict(features)
df['Spectral_Cluster'] = spectral_labels
# Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Annual_Income (£K)', y='Spending_Score',
                hue='Spectral_Cluster', palette='Spectral')
plt.title('Spectral Clustering')
plt.legend()
plt.show()
# Silhouette Score
spectral_score = silhouette_score(features, spectral_labels)
print(f"Silhouette Score for Spectral: {spectral_score:.3f}")