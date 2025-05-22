GNN-Based Anomaly Detection in Telecommunications Data
An advanced, end-to-end pipeline for detecting anomalies in mobile network user behavior using Graph Neural Networks (GNNs), traditional machine learning, and graph-based statistics.


This repository implements a graph-based anomaly detection system tailored for telecom datasets containing per-user session and network activity. It combines GNN autoencoders, Isolation Forests, and statistical graph methods to detect abnormal MSISDN behavior across thousands of features and sessions.

Core Components:-

🔹 Data Preprocessing & Feature Engineering
Aggregate total upload/download/session duration across 2G/3G/4G/5G

Calculate usage/session and zero-KB session ratios

Identify dominant network type

Normalize features using StandardScaler

🔹 Graph Construction
Nodes: MSISDN activity entries

Edges:

Same CUSTOMER_ID: weight = 0.9

Cosine similarity > 0.95 between feature vectors

Format: PyTorch Geometric Data object

🔹 GNN Autoencoder
Architecture: Two-layer encoder/decoder with GCNConv

Trained to reconstruct node features

Nodes with high reconstruction loss are flagged as anomalies

🔹 Isolation Forest
Includes graph centrality metrics: degree, PageRank, eigenvector centrality

Trained on combined feature vector

🔹 Statistical Neighborhood Anomaly
Compare each node to its neighbors using:

Euclidean & Manhattan distance

Local density score

Aggregated score compared to 95th percentile threshold

Overall Anomaly Statistics
Each method identified 5 anomalies, corresponding to an anomaly rate of 5% across the dataset.

Method          	Anomalies	  Anomaly Rate	  Threshold (95th Percentile)
GNN AutoEncoder	  5	          5.0%	          1.89
Isolation Forest	5	          5.0%	          0.60
Statistical Graph	5	          5.0%	          13.06
