# E-commerce-Clustering-Models-for-Targeted-Marketing
The project compares different clustering models for Targeted Marketing of E-commerce business
## Introduction:
This assignment involves developing and evaluating clustering models for an e-commerce dataset in order to explore meaningful customer segments that can guide a targeted marketing strategy for an e-commerce company aiming to boost profitability. By analysing customer demographics and purchasing behaviour, this investigation explores multiple clustering models which includes K-means, hierarchical clustering, DBSCAN and spectral clustering to identify the most effective method (Tan, Steinbach & Kumar, 2019; Jain, 2010). In addition to providing comparative study of the clustering model’s performance, the analysis also delivers actionable insights aligned with the company’s strategic marketing goals.
## Methodology:
This task follows a structured methodology in order to explore customer segmentation using clustering techniques. The overall workflow includes the following stages:

### 1. Data Understanding and Exploration
The dataset was first loaded and explored to understand its structure and contents. It included customer-related features such as Age, Annual Income (£K), Spending Score, and Gender (originally wrongly labelled as 'Genre'). Key exploratory steps taken were as follows:
•	Identifying data types and basic statistics.
•	Checking for missing values and duplicated records(Han, Pei & Kamber, 2011).
•	Visualizing the feature distributions in order to detect skewness or outliers(McKinney, 2017).

### 2. Data Cleaning and Preprocessing
a. Handling Missing Values
b. Handling Duplicates
c. Correction of Column Labels
d. Encoding Categorical Variables
e. Feature Standardization (Kuhn & Johnson, 2013; James et al., 2021)

### 3. Clustering Models
To identify meaningful customer segments, the following clustering algorithms were implemented:
a. K-Means Clustering
•	The Elbow Method for determining optimal number of clusters based on within-cluster sum of squares (SSE) (Kodinariya & Makwana, 2013).
•	Silhouette Score analysis to further validated the cluster separation (Rousseeuw, 1987).
•	The number of clusters was selected based on the above two metrics, and final K-Means clustering was applied.
•	Clusters were visualized using scatter plots colored by cluster label.
b. Hierarchical Clustering
•	A Dendrogram was plotted to visualize cluster merging and to determine a suitable number of clusters(Murtagh & Contreras, 2012).
•	Agglomerative Clustering was used to plot and visualised using scatter plot.
c. DBSCAN
•	DBSCAN, a density-based clustering algorithm, was used to capture non-spherical and arbitrarily shaped clusters (Ester et al., 1996).
•	The main Parameters eps and min_samples were tuned manually.
d. Spectral Clustering
•	Spectral Clustering was implemented and cluster assignments were visualized similarly to the other methods using scatter plot (Ng, Jordan & Weiss, 2002).

### 4. Cluster Evaluation and Interpretation
Each clustering model was evaluated based on:
•	Compactness (how well data points were grouped within a cluster).
•	Separation (how distinct the clusters are from each other).
•	Silhouette Score (a measure of how well each data point fits its assigned cluster).

### 5. Visualization
A variety of plots were generated to support interpretation:
•	Elbow and Silhouette plots for K-Means.
•	Scatter plots for all clustering results.
•	Dendrogram for Hierarchical Clustering to identify optimum no. of clusters.
•	Cluster profiling using bar charts to show average feature values per cluster.

## Exploratory Data Analysis
### Data Import 
The dataset was first loaded in a dataframe and explored to understand its structure and contents. The features that the dataset includes are Age, Annual Income (£K), Spending Score, and Gender (originally labeled as 'Genre'). 

 ![image](https://github.com/user-attachments/assets/cf019468-c772-41b4-9ff8-862d4ea5f8b6)

Figure 1: First Five rows of the original datset

A view of first five rows of the raw dataset can be seen in Figure 1. It has a total of four features Genre(spelling error, it is supposed to be Gender), Age, Annual_Income and Spending score.

### Data Cleaning and Preprocessing
#### a. Handling of Missing Values
![image](https://github.com/user-attachments/assets/b11779d6-28fd-4123-8c94-0d3088812875)
 
Figure 2: Missing Values Table for each column

Missing values were assessed using “df.isnull().sum()” and visually confirmed that there are a total of 59 missing values, 25 in ‘Age’ column, 19 in ‘Annual Income (£K)’ and 15 in ‘Spending Score’. Based on further exploratory analysis, missing values were dropped from the dataset as there was no logical reason for imputation of the missing values and the number of missing values is not significantly high. Details of data distribution is discussed below:

![image](https://github.com/user-attachments/assets/0ca25234-e8bf-44b8-a8b8-2b7365a1e57e)

Figure 3: Distribution of Age alongside Box Plot for Outliers

In Figure 3 above, the distribution of age depicts a fairly balanced behaviour, while the boxplot displays no significant outliers and Inter quartile range from 29 to 55 and median being 42. As mentioned earlier, the rows with missing values were dropped as there is no logical reasoning for imputing/replacing the missing values.

 ![image](https://github.com/user-attachments/assets/4b2f2a2d-52a0-4c2f-bb6a-8e2638e127f0)

Figure 4: Distribution of 'Annual_Income (£K) alongside Boxplot for Outliers

According to Figure 4, the data is slightly right skewed and Boxplot displays high income outliers on the upper end. Therefore, interpolating the missing values with any logical calculation seems to be a bit of stretch as the quantity is not significant..

![image](https://github.com/user-attachments/assets/1e5c813c-f10d-4a24-a3c5-07d4aeeda44c)

Figure 5: Distribution of 'Spending Score' alongside Boxplot for Outliers

The data in Figure 5, depicts a different story as the distribution implies no significant skewness, but it displays two different peaks around low 5-10 and high 50-60. Although the distribution cannot be said to be absolutely normally distributed, but it is evenly distributed. In the boxplot on the right, the median being around 35 and the interquartile range being from 20 to 50, it is safe to say that there no extreme outliers. Although, both mean and median are safe choices for imputation here, but again imputations lacks any logical explanation due to lack of more features in the dataset.

#### b. Handling Duplicates
![image](https://github.com/user-attachments/assets/2ff96258-abdd-4d71-bcdb-396441549e03)
 
Figure 6: No. of Duplicate rows

As seen in Figure 6, duplicate records were checked using ‘.duplicated()’ and a total of 252 fully identical rows were found, as there might be a sea of reasons for these duplicates such as redundant data entries or multiple purchases by the same customer. The lack of a unique customer ID or purchase ID in the dataset leaves us with incomplete information. Therefore, the duplicated rows have been retained for this model.

#### c. Correcting Column Label
![image](https://github.com/user-attachments/assets/5dafecb7-38b4-4560-91ea-409fcb7bbf47)

Figure 7: Renamed 'Gender' Column

The column labelled "Genre" was identified as actually representing Gender. Therefore, this column was renamed accordingly for clarity.

#### d. Encoding Categorical Variable
![image](https://github.com/user-attachments/assets/fc8d17dd-470a-412f-b2f1-93adc2b2f6f2)

Figure 8: Label Encoded Gender column

Furthermore, The ‘Gender’ column was label encoded using LabelEncoder, converting the string categories (“Male”, “Female”) into numeric format (0, 1), which is suitable for machine learning models.

#### e. Feature Scaling
![image](https://github.com/user-attachments/assets/3284ee26-e33e-4dc2-a590-8d9f7d4c32ac)
 
Figure 9: Feature Scaling using Standard scaler()

Numerical features (Age, Annual Income (£K), Spending Score) were normalized using StandardScaler method to ensure equal weighting during clustering. The Gender column, now numeric after label encoding, was also included in the scaled dataset and repositioned to maintain consistency in analysis.

### Clustering Models and Observations
To identify meaningful customer segments, the following clustering algorithms were implemented:
#### a. K-Means Clustering
•	Using the elbow method optimal number of clusters was identified.

![image](https://github.com/user-attachments/assets/8f316a62-f9a0-4718-891c-4dd9896523a4)

Figure 10: Elbow method for k value for optimal number of clusters

•	Silhouette Score analysis further validates the cluster separation quality.
![image](https://github.com/user-attachments/assets/4534dcaa-4bc6-4a76-9361-31b3fcba82fc)
 
Figure 11: Silhouette score for K-means

The best number of clusters, which is six clusters was selected based on the two metrics mentioned above, and final K-Means clustering was applied. Clusters were visualized using scatter plots coloured by cluster label.

![image](https://github.com/user-attachments/assets/1a60871b-47c1-408c-bd7e-804f75ff2bac)

Figure 12: K-Means Clusters

In Figure 12, each colour represents distinct group of customers with similar income and spending scores. There are a total of six clusters which are represented by six different colours. The 0th cluster represents the group with high income and low spending; the 1st group represents high income and high spending customers; the 2nd group represents low income and low spending customers; the 3rd group represents mid to high income and mid to high spending customers; the 4th group low income and high spending customers; the 5th group represents low to mid income and low spending customers. 

## Cluster Profile Summary:

![image](https://github.com/user-attachments/assets/a2d4ff62-1d3c-4c45-a996-ee995d470768)

Figure 13: Cluster Profile Summary

A detailed statistical summary can be seen in the above figure 13, which describes the percentages males or females that belong to a particular cluster in addition to the average income and spending score of that particular group.
•	Number of customers in each cluster

![image](https://github.com/user-attachments/assets/0a43b757-af2c-474c-bef7-a711cba0b9d8)

Figure 14: No. of customers per clusters

Figure 14 visualizes the number of customers per cluster with simple bar plot. It can be seen that the cluster 2 has the highest number of customers with 140+ customers while the 4th clusters comprises of 135 customers. The 1st and 3rd clusters have almost similar number of customers with approximately 130 customers. The 5th cluster has just above 100 customers while the 0th cluster consists lowest percentage of customers with just above 90 customers.

#### b. Hierarchical Clustering
•	A Dendrogram was plotted using Ward linkage to visualize cluster merging and determine a suitable cut-off for the number of clusters. The number of clusters according to the dendrogram was found to be six.

![image](https://github.com/user-attachments/assets/29ce574c-8b97-4815-b22b-6fa5417e54d2)

Figure 15: Hierarchical Dendrogram

•	The model was fitted using Agglomerative Clustering, and results were visualized with scatter plots. Details of which are as follows:
 
![image](https://github.com/user-attachments/assets/72918908-cb8b-4957-934b-d7b5af155c63)

Figure 16: Scatter plot of Hierarchical Cluster

According to the results in Figure 16, Hierarchical clusters also found a similar number of clusters, i.e. six. The 0th cluster is low income and high spending customers; the 1st  cluster comprises of high income and low spending customers; the 2nd cluster comprises of mid to high income and low spending customers; the 3rd cluster comprises of mid to high income and high spending customers; the 4th cluster comprises of low income and low spending customers while the 5th cluster comprises high income and high spending customers.

#### c. DBSCAN

![image](https://github.com/user-attachments/assets/b7d546fe-2beb-4880-b714-f10139eac5c6)

Figure 17: DBSCAN Clusters

The clusters in Figure 17 display weird clusters most likely due to wrongly label points as outliers or merging small gaps as one cluster. This can be mainly due to the fact that DBSCAN works best with data with high density variation (Ester et al., 1996), i.e. irregular and noisy dataset. Whereas, the data in this project is clean and has no complex shapes or noise.

#### d. Spectral Clustering
 
![image](https://github.com/user-attachments/assets/02c772f5-83f0-48ad-bbf6-31bdf922468a)

Figure 18: Scatter Plot for Spectral Clustering

The scatter plot in Figure 18 does not show a clear distinct clusters mainly because of the lack of feature in the dataset. This is also proven by the silhouette score of 0.105 which is closer to 0 and indicates overlapping clusters.
