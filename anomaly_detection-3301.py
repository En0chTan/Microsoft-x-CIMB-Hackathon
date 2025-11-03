#!/usr/bin/env python
# coding: utf-8

# ## anomaly_detection-3301
# 
# New notebook

# In[2]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install seaborn


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from IPython.display import display
from pyspark.sql import SparkSession

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Load the dataset
df = pd.read_csv('abfss://hack_group3@onelake.dfs.fabric.microsoft.com/lakehouse_group3.Lakehouse/Files/bank_transactions_data_2.csv') 

# Display basic information about the dataset
print("Shape of the dataset:", df.shape)
display(df.head())


# 

# In[5]:


# Select features for clustering
features = ['TransactionAmount', 'TransactionDuration'] 
X = df[features].copy()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 10)  # Test for clusters from 1 to 10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


# In[6]:


# Plot the Elbow Method
plt.figure(figsize=(12, 6))
plt.plot(K, inertia, marker='o', linestyle='-', color=sns.color_palette("YlOrBr", 10)[8])  
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, linestyle='--')  
plt.show()


# In[ ]:


# Fit K-means with the chosen number of clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# Assign clusters and calculate distance to cluster centroid
df['Cluster'] = kmeans.labels_
df['DistanceToCentroid'] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)

# Identify potential frauds based on distance threshold
threshold = df['DistanceToCentroid'].quantile(0.95)
potential_frauds = df[df['DistanceToCentroid'] > threshold]

print(f"Number of potential frauds detected: {len(potential_frauds)}")
df[potential_frauds](1000)


# In[31]:


# Assuming 'TransactionID' and 'DistanceToCentroid' are valid columns in potential_frauds
# Assuming 'potential_frauds' is your Pandas DataFrame

# 1. Correctly select the TransactionID column from the potential_frauds DataFrame
# This results in a Pandas Series (which has the .tolist() method)
id_series = potential_frauds['TransactionID']

# 2. Convert the Pandas Series to a standard Python list
list_of_ids = id_series.tolist()

# 3. Print the list to view the IDs
print("--- List of Potential Fraud Transaction IDs ---")
print(list_of_ids)

# Optional: Print the total count to confirm
print(f"\nTotal IDs extracted: {len(list_of_ids)}")

# # 2. Select these specific columns from the potential_frauds DataFrame
# df_to_export = potential_frauds[columns_to_show]

# # Optional: Display the DataFrame that will be exported (only TransactionID and DistanceToCentroid)
# display(df_to_export) 

# # Define the file path and name
# output_file = 'potential_frauds_summary.csv' 

# # 3. Export the selected DataFrame to CSV
# df_to_export.to_csv(output_file, index=False)

# print(f"Summary data successfully exported to: {output_file}")


# In[26]:


pd.set_option('display.max_row',none)
print(potential_frauds)


# In[8]:


# Define the file path and name
output_file = 'potential_frauds_output.csv'

# Export the DataFrame to CSV
potential_frauds.to_csv(output_file, index=False)


# In[9]:


# Visualize clusters and potential frauds (2D plot for simplicity with legend)
plt.figure(figsize=(12, 6))

cluster_colors = sns.color_palette("YlOrBr", n_colors=len(np.unique(kmeans.labels_)))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=[cluster_colors[i] for i in kmeans.labels_], alpha=0.5, label='Cluster Points')
centroids = plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')

frauds = plt.scatter(X_scaled[potential_frauds.index, 0], X_scaled[potential_frauds.index, 1], c='black', label='Potential Frauds', edgecolors='k')

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('K-means Clustering and Potential Frauds')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()


# In[ ]:


# Map the cluster labels to descriptive names
label_mapping = {
    -1: 'Fraud (Outliers)',  
    0: 'Normal',
    1: 'Suspicious Group 1',
    2: 'Suspicious Group 2',
    3: 'Suspicious Group 3',
    4: 'Suspicious Group 4',
}

# Select relevant features for DBSCAN 
features = ['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'LoginAttempts']
X = df[features].copy()

X = X.fillna(X.mean())

# Standardize the features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)  
dbscan.fit(X_scaled)

df['DBSCAN_Cluster'] = dbscan.labels_

# Map cluster labels to descriptive names
df['Cluster_Description'] = df['DBSCAN_Cluster'].map(label_mapping)

# Identify outliers (noise points) labeled as -1
potential_frauds = df[df['DBSCAN_Cluster'] == -1]
print(f"Number of potential frauds detected by DBSCAN: {len(potential_frauds)}")
display(potential_frauds.show(1000))


# In[11]:


# Visualize clusters and potential frauds
plt.figure(figsize=(10, 6))
unique_labels = np.unique(dbscan.labels_)

cluster_colors = sns.color_palette("YlOrBr", n_colors=len(unique_labels)) 

colors = []
for i, k in enumerate(unique_labels):
    if k == -1:
        colors.append([0, 0, 0, 1])  # Black for noise
    else:
        colors.append(cluster_colors[i % len(cluster_colors)])  

for k, col in zip(unique_labels, colors):
    class_member_mask = (dbscan.labels_ == k)
    xy = X_scaled[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], color=col, edgecolor='k', alpha=0.7, label=label_mapping.get(k, f'Cluster {k}'))

plt.title('DBSCAN - Clusters and Potential Frauds (Outliers)')
plt.xlabel(features[0])  # TransactionAmount
plt.ylabel(features[1])  # TransactionDuration
plt.legend()
plt.grid(True, linestyle='--')  
plt.show()


# In[12]:


# Define outlier mapping
outlier_mapping = {1: 'Normal', -1: 'Potential Fraud'}

# Select relevant features for fraud detection
features = ['TransactionAmount', 'TransactionDuration', 'AccountBalance', 'LoginAttempts']  # Modify as needed
X = df[features].copy()

X = X.fillna(X.mean())

# Standardize the features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_scaled)

# Predict anomalies
df['AnomalyScore'] = iso_forest.decision_function(X_scaled)
df['IsAnomaly'] = iso_forest.predict(X_scaled)  

# Map results to descriptive labels
df['AnomalyLabel'] = df['IsAnomaly'].map(outlier_mapping)

# Filter out detected anomalies
potential_frauds = df[df['IsAnomaly'] == -1]
print(f"Number of potential frauds detected: {len(potential_frauds)}")
display(potential_frauds.head())


# In[13]:


potential_frauds.shape


# In[14]:


cmap = plt.get_cmap('YlOrBr')
colors = np.array([cmap(0.9) if anomaly == -1 else cmap(0.3) for anomaly in df['IsAnomaly']])

# Visualize potential frauds (TransactionAmount vs AccountBalance)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_scaled[:, 0],  # TransactionAmount 
    X_scaled[:, 2],  # AccountBalance 
    c=colors, 
    alpha=0.7, 
    edgecolors='k', 
    label='Data Points'
)

normal_patch = mpatches.Patch(color=cmap(0.3), label='Normal')
fraud_patch = mpatches.Patch(color=cmap(0.9), label='Potential Fraud')
plt.legend(handles=[normal_patch, fraud_patch], title='Outlier Prediction')

plt.title('Isolation Forest - Potential Anomalies')
plt.xlabel(features[0])  
plt.ylabel(features[2])  

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[ ]:


# 'spark' is the pre-configured SparkSession in Fabric Notebooks
spark_df = spark.createDataFrame(potential_frauds)

# Optional: Display the PySpark DataFrame to verify
spark_df.show(400)

