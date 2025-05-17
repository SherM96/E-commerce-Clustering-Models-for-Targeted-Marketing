# Analysis Report

Prepared by: SherM96

Task: Evaluation and Recommendation of Clustering Techniques

Objective: Identify meaningful customer segments based on purchasing behaviour to guide strategic marketing and sales initiatives.

## Objective & Motivation
The aim of this task was to apply Machine Learning techniques to discover customer groups based on two key behaviours:
•	Annual Income (£K)
•	Spending Score 
Analysing these two segments will allow the business to target marketing efforts more precisely, develop personalized promotions, improve customer satisfaction and retention.

## Investigation Process
Four Clustering Algorithm were tested on the dataset:
1.	K-Means Clustering
Groups customers by minimizing distance from cluster centers. Best for well-separated, round-shaped clusters (Jain, 2010).
2.	Hierarchical Clustering
Builds a tree-like structure (dendrogram) based on similarity between customers. Doesn’t need a predefined number of clusters (Murtagh & Contreras, 2012).
3.	DBSCAN
Detects clusters of varying shapes and finds outliers based on density (Ester et al., 1996).
4.	Spectral Clustering
Uses graph theory to cluster complex data structures. Effective for non-linear or non-convex patterns (Ng, Jordan & Weiss, 2002).

## Summary of Results
 K-Means Clustering
•	A total of 6 clusters were formed.
•	Visualization showed clearly separated groups/clusters of customers.
•	Seemed ideal for our numeric data (Income, Spending Score).
•	Easy to interpret for action planning.
 Hierarchical Clustering
•	Dendrogram helped reveal natural grouping of 6 clusters as well.
•	This seemed effective for small datasets such as the one we used.
•	Less scalable and slower than K-Means for larger customer bases.

 DBSCAN
•	Unable to clearly segment the data.
•	Misclassified several points due to similar density.
•	Not suitable due to lack of noise/outliers in our data.
•	Lack of features in dataset was noticed for application of DBSCAN.

 Spectral Clustering
•	Gave reasonable clusters but added unnecessary complexity.
•	Computationally heavier. Usually used for datasets with high number of features.
•	No added value over K-Means for our simple, structured data.

## Final Evaluation & Recommendation
Recommended Model: 
K-Means Clustering
Reasons:
•	Produced well-defined and understandable customer clusters.
•	It matches the business case as the customer groups are shaped clearly by income and spending patterns.
•	It is fast and scalable and can be applied easily as more customer data is collected.
•	Outperformed other models in terms of clarity, practicality, and speed.

## Key Insights
There are a total of six clusters: The 0th cluster represents the group with high income and low spending; the 1st group represents high income and high spending customers; the 2nd group represents low income and low spending customers; the 3rd group represents mid to high income and mid to high spending customers; the 4th group low income and high spending customers; the 5th group represents low to mid income and low spending customers. 

Cluster | Income_Status |	Spending_Status |	Customer_Description
------- | ------------- | ----------------| --------------------
0	      | High          |	Low             |	High income, low spending
1       |	High          |	High            |	High income, high spending
2       |	Low           |	Low             |	Low income, low spending
3       |	Mid to High   |	Mid to High     |	Balanced earners and spenders
4       |	Low           |	High            |	Low income, high spending
5       |	Low to mid    |	Low             |	Budget-conscious shoppers



## Conclusion & Future Recommendations

•	K-Means is the optimal clustering solution for our data and business needs.
•	Clear segmentation will enable tailored marketing, loyalty programs, and promotions.
•	Periodic re-clustering should be performed as more and more customer data is gathered.
•	Consider expanding the model to include age, location, and purchase frequency for deeper insights.
