import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
import pickle
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class TelecomGNNAnomalyDetector:
    def __init__(self, data_path=None, model_name="telecom_gnn_anomaly_detector"):
        """
        Initialize the GNN-based anomaly detector for telecommunications data
        """
        self.data_path = data_path
        self.model_name = model_name
        self.raw_data = None
        self.processed_data = None
        self.graph_data = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.anomalies = None
        self.models = {}
        self.performance_metrics = {}
        self.anomaly_report = {}
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the telecommunications data
        """
        print("Loading and preprocessing data...")
        
        # Load data
        self.raw_data = pd.read_csv(self.data_path)
        
        # Handle scientific notation in MSISDN
        self.raw_data['MSISDN'] = self.raw_data['MSISDN'].astype(str)
        
        # Convert date
        self.raw_data['S_DATE'] = pd.to_datetime(self.raw_data['S_DATE'])
        
        # Feature engineering
        self.processed_data = self._engineer_features()
        
        print(f"Data loaded: {len(self.processed_data)} records")
        print(f"Features: {self.feature_columns}")
        
    def _engineer_features(self):
        """
        Engineer features for anomaly detection
        """
        df = self.raw_data.copy()
        
        # Calculate totals across technologies
        df['total_upload'] = (df['UPLOAD_2G'].fillna(0) + df['UPLOAD_3G'].fillna(0) + 
                             df['UPLOAD_4G'].fillna(0) + df['UPLOAD_5G'].fillna(0))
        df['total_download'] = (df['DOWNLOAD_2G'].fillna(0) + df['DOWNLOAD_3G'].fillna(0) + 
                               df['DOWNLOAD_4G'].fillna(0) + df['DOWNLOAD_5G'].fillna(0))
        df['total_duration'] = (df['DURATION_2G'].fillna(0) + df['DURATION_3G'].fillna(0) + 
                               df['DURATION_4G'].fillna(0) + df['DURATION_5G'].fillna(0))
        
        # Ratio features
        df['upload_download_ratio'] = np.where(df['total_download'] > 0, 
                                              df['total_upload'] / df['total_download'], 0)
        df['data_per_session'] = np.where(df['TOT_SESSIONS'] > 0,
                                         (df['total_upload'] + df['total_download']) / df['TOT_SESSIONS'], 0)
        df['duration_per_session'] = np.where(df['TOT_SESSIONS'] > 0,
                                             df['total_duration'] / df['TOT_SESSIONS'], 0)
        
        # Zero session ratios
        df['zero_kb_ratio'] = np.where(df['TOT_SESSIONS'] > 0,
                                      df['TOTAL_SESSIONS_0KB'].fillna(0) / df['TOT_SESSIONS'], 0)
        
        # Technology preference (dominant technology)
        tech_cols = ['UPLOAD_2G', 'UPLOAD_3G', 'UPLOAD_4G', 'UPLOAD_5G']
        df['dominant_tech'] = df[tech_cols].fillna(0).idxmax(axis=1).str.extract(r'(\d)G').astype(int)
        
        # Time-based features
        df['hour'] = df['S_DATE'].dt.hour
        df['day_of_week'] = df['S_DATE'].dt.dayofweek
        df['month'] = df['S_DATE'].dt.month
        
        # Session distribution features
        session_cols = ['TOTAL_SESSIONS_0KB', 'TOTAL_SESSIONS_1KB', 
                       'TOTAL_SESSIONS_100KB', 'TOTAL_SESSIONS_300KB']
        for col in session_cols:
            df[f'{col}_ratio'] = np.where(df['TOT_SESSIONS'] > 0,
                                         df[col].fillna(0) / df['TOT_SESSIONS'], 0)
        
        # Data efficiency metrics
        df['upload_efficiency'] = np.where(df['total_duration'] > 0,
                                          df['total_upload'] / df['total_duration'], 0)
        df['download_efficiency'] = np.where(df['total_duration'] > 0,
                                            df['total_download'] / df['total_duration'], 0)
        
        # Session behavior patterns
        df['avg_session_size'] = np.where(df['TOT_SESSIONS'] > 0,
                                         (df['total_upload'] + df['total_download']) / df['TOT_SESSIONS'], 0)
        df['session_diversity'] = (df['TOTAL_SESSIONS_1KB'].fillna(0) > 0).astype(int) + \
                                 (df['TOTAL_SESSIONS_100KB'].fillna(0) > 0).astype(int) + \
                                 (df['TOTAL_SESSIONS_300KB'].fillna(0) > 0).astype(int)
        
        # Select relevant features for modeling
        self.feature_columns = [
            'total_upload', 'total_download', 'total_duration',
            'upload_download_ratio', 'data_per_session', 'duration_per_session',
            'zero_kb_ratio', 'dominant_tech', 'TOT_SESSIONS', 'OPEN_CNT',
            'TOTAL_SESSIONS_0KB_ratio', 'TOTAL_SESSIONS_1KB_ratio',
            'TOTAL_SESSIONS_100KB_ratio', 'TOTAL_SESSIONS_300KB_ratio',
            'upload_efficiency', 'download_efficiency', 'avg_session_size',
            'session_diversity', 'hour', 'day_of_week', 'month'
        ]
        
        return df[self.feature_columns + ['MSISDN', 'S_DATE', 'CUSTOMER_ID']].fillna(0)
    
    def create_graph_structure(self, similarity_threshold=0.7):
        """
        Create graph structure based on customer relationships and feature similarity
        """
        print("Creating graph structure...")
        
        # Normalize features
        node_features = self.scaler.fit_transform(self.processed_data[self.feature_columns])
        
        # Create edges based on multiple criteria
        edges = []
        edge_weights = []
        num_nodes = len(self.processed_data)
        
        print(f"Building edges for {num_nodes} nodes...")
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = 0
                
                # Customer-based edges (same customer)
                if self.processed_data.iloc[i]['CUSTOMER_ID'] == self.processed_data.iloc[j]['CUSTOMER_ID']:
                    weight = 0.9
                    edges.append([i, j])
                    edge_weights.append(weight)
                else:
                    # Feature similarity edges
                    similarity = self._calculate_similarity(node_features[i], node_features[j])
                    if similarity > similarity_threshold:
                        weight = similarity
                        edges.append([i, j])
                        edge_weights.append(weight)
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1) if edge_weights else torch.empty((0, 1), dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        
        self.graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"Graph created: {num_nodes} nodes, {len(edges)} edges")
        print(f"Average node degree: {len(edges) * 2 / num_nodes:.2f}")
        
    def _calculate_similarity(self, feat1, feat2, method='cosine'):
        """
        Calculate similarity between two feature vectors
        """
        if method == 'cosine':
            dot_product = np.dot(feat1, feat2)
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return dot_product / (norm1 * norm2)
        elif method == 'euclidean':
            return 1 / (1 + np.linalg.norm(feat1 - feat2))
        else:
            return np.corrcoef(feat1, feat2)[0, 1] if not np.isnan(np.corrcoef(feat1, feat2)[0, 1]) else 0

    def save_model(self, save_path=None):
        """
        Save the trained models and preprocessing components
        """
        if save_path is None:
            save_path = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Saving model to {save_path}...")
        
        # Create save dictionary
        save_dict = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'models': self.models,
            'performance_metrics': self.performance_metrics,
            'anomaly_report': self.anomaly_report,
            'graph_structure_params': {
                'similarity_threshold': 0.7  # Store for reuse
            }
        }
        
        # Save with pickle
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Save performance metrics as JSON
        with open(f"{save_path}_metrics.json", 'w') as f:
            json.dump(convert_numpy_types(self.performance_metrics), f, indent=2)
        
        print(f"Model saved successfully as {save_path}.pkl")
        return save_path
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        """
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.scaler = save_dict['scaler']
        self.feature_columns = save_dict['feature_columns']
        self.models = save_dict['models']
        self.performance_metrics = save_dict.get('performance_metrics', {})
        self.anomaly_report = save_dict.get('anomaly_report', {})
        
        print("Model loaded successfully!")
        return save_dict.get('graph_structure_params', {})
    
    def predict_new_data(self, new_data_path, model_path=None):
        """
        Use trained model to detect anomalies in new data
        """
        print("=== PREDICTING ON NEW DATA ===")
        
        if model_path:
            graph_params = self.load_model(model_path)
        
        # Load and preprocess new data
        old_data_path = self.data_path
        self.data_path = new_data_path
        self.load_and_preprocess_data()
        
        # Create graph structure with same parameters
        similarity_threshold = 0.7  # Use same threshold as training
        self.create_graph_structure(similarity_threshold)
        
        # Run anomaly detection with loaded models
        results = {}
        
        if 'gnn_autoencoder' in self.models:
            print("Running GNN AutoEncoder on new data...")
            results['GNN_AutoEncoder'] = self._predict_gnn_autoencoder()
        
        if 'isolation_forest' in self.models:
            print("Running Isolation Forest on new data...")
            results['Isolation_Forest'] = self._predict_isolation_forest()
        
        if 'statistical_graph' in self.models:
            print("Running Statistical Graph method on new data...")
            results['Statistical_Graph'] = self._predict_statistical_graph()
        
        # Generate prediction report
        prediction_report = self.generate_prediction_report(results)
        
        # Restore original data path
        self.data_path = old_data_path
        
        return results, prediction_report

class GNNAutoEncoder(nn.Module):
    """
    Graph Neural Network AutoEncoder for anomaly detection
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(GNNAutoEncoder, self).__init__()
        
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.conv3 = GCNConv(latent_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        x = F.relu(self.conv3(z, edge_index))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        return x
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_reconstructed = self.decode(z, edge_index)
        return x_reconstructed, z

def calculate_performance_metrics(anomaly_scores, labels=None, threshold_percentile=95):
    """
    Calculate comprehensive performance metrics
    """
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    predicted_anomalies = anomaly_scores > threshold
    
    metrics = {
        'threshold': threshold,
        'num_anomalies': int(predicted_anomalies.sum()),
        'anomaly_rate': float(predicted_anomalies.mean() * 100),
        'threshold_percentile': threshold_percentile,
        'score_statistics': {
            'mean': float(np.mean(anomaly_scores)),
            'std': float(np.std(anomaly_scores)),
            'min': float(np.min(anomaly_scores)),
            'max': float(np.max(anomaly_scores)),
            'median': float(np.median(anomaly_scores)),
            'q25': float(np.percentile(anomaly_scores, 25)),
            'q75': float(np.percentile(anomaly_scores, 75))
        }
    }
    
    # If we have labels (for validation), calculate additional metrics
    if labels is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            metrics['precision'] = float(precision_score(labels, predicted_anomalies))
            metrics['recall'] = float(recall_score(labels, predicted_anomalies))
            metrics['f1_score'] = float(f1_score(labels, predicted_anomalies))
            metrics['roc_auc'] = float(roc_auc_score(labels, anomaly_scores))
        except:
            metrics['supervised_metrics'] = "Not available - insufficient label variation"
    
    return metrics

def run_anomaly_detection_methods(detector):
    """
    Run multiple anomaly detection methods and calculate performance metrics
    """
    methods_results = {}
    
    # Method 1: GNN AutoEncoder
    print("\n=== Running GNN AutoEncoder ===")
    autoencoder_results = run_gnn_autoencoder(detector)
    methods_results['GNN_AutoEncoder'] = autoencoder_results
    
    # Method 2: Isolation Forest with Graph Features
    print("\n=== Running Isolation Forest with Graph Features ===")
    isolation_results = run_isolation_forest_graph(detector)
    methods_results['Isolation_Forest'] = isolation_results
    
    # Method 3: Graph-based Statistical Anomaly Detection
    print("\n=== Running Graph-based Statistical Anomaly Detection ===")
    statistical_results = run_statistical_graph_anomaly(detector)
    methods_results['Statistical_Graph'] = statistical_results
    
    # Store performance metrics
    detector.performance_metrics = {}
    for method, results in methods_results.items():
        detector.performance_metrics[method] = calculate_performance_metrics(results['anomaly_scores'])
    
    return methods_results

def run_gnn_autoencoder(detector):
    """
    Run GNN AutoEncoder for anomaly detection with performance tracking
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    input_dim = detector.graph_data.x.shape[1]
    model = GNNAutoEncoder(input_dim, hidden_dim=64, latent_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    # Move data to device
    data = detector.graph_data.to(device)
    
    # Training with loss tracking
    model.train()
    losses = []
    
    print("Training GNN AutoEncoder...")
    for epoch in range(1000):
        optimizer.zero_grad()
        x_reconstructed, z = model(data.x, data.edge_index)
        loss = F.mse_loss(x_reconstructed, data.x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    # Store trained model
    detector.models['gnn_autoencoder'] = {
        'model': model.cpu(),
        'training_losses': losses,
        'input_dim': input_dim
    }
    
    # Anomaly detection
    model.eval()
    data = data.cpu()
    with torch.no_grad():
        x_reconstructed, z = model(data.x, data.edge_index)
        reconstruction_errors = F.mse_loss(x_reconstructed, data.x, reduction='none').mean(dim=1)
        
        # Calculate anomaly threshold (95th percentile)
        threshold = torch.quantile(reconstruction_errors, 0.95)
        anomalies = (reconstruction_errors > threshold).numpy()
        anomaly_scores = reconstruction_errors.numpy()
    
    results = {
        'anomalies': anomalies,
        'anomaly_scores': anomaly_scores,
        'threshold': threshold.item(),
        'num_anomalies': anomalies.sum(),
        'anomaly_rate': anomalies.mean() * 100,
        'training_losses': losses,
        'final_loss': losses[-1]
    }
    
    print(f"GNN AutoEncoder detected {results['num_anomalies']} anomalies ({results['anomaly_rate']:.2f}%)")
    print(f"Final training loss: {results['final_loss']:.6f}")
    
    return results

def run_isolation_forest_graph(detector):
    """
    Run Isolation Forest with graph-enhanced features
    """
    # Extract node features and graph statistics
    node_features = detector.graph_data.x.numpy()
    
    # Calculate graph-based features
    G = to_networkx(detector.graph_data, to_undirected=True)
    
    print("Calculating graph centrality measures...")
    # Node centrality measures
    degree_centrality = list(nx.degree_centrality(G).values())
    betweenness_centrality = list(nx.betweenness_centrality(G, k=min(100, len(G))).values())
    clustering_coeff = list(nx.clustering(G).values())
    
    # PageRank and eigenvector centrality
    try:
        pagerank = list(nx.pagerank(G, max_iter=100).values())
        eigenvector_centrality = list(nx.eigenvector_centrality(G, max_iter=100).values())
    except:
        pagerank = [0] * len(G)
        eigenvector_centrality = [0] * len(G)
    
    # Combine original features with graph features
    graph_features = np.column_stack([
        node_features,
        degree_centrality,
        betweenness_centrality,
        clustering_coeff,
        pagerank,
        eigenvector_centrality
    ])
    
    # Run Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.05, 
        random_state=42, 
        n_estimators=200,
        max_samples='auto',
        bootstrap=True
    )
    
    print("Training Isolation Forest...")
    anomalies = iso_forest.fit_predict(graph_features)
    anomaly_scores = iso_forest.score_samples(graph_features)
    
    # Store trained model
    detector.models['isolation_forest'] = {
        'model': iso_forest,
        'graph_feature_names': ['degree_centrality', 'betweenness_centrality', 
                               'clustering_coeff', 'pagerank', 'eigenvector_centrality']
    }
    
    # Convert to binary (1 for normal, -1 for anomaly)
    anomalies_binary = (anomalies == -1)
    
    results = {
        'anomalies': anomalies_binary,
        'anomaly_scores': -anomaly_scores,  # Negative for consistency (higher = more anomalous)
        'num_anomalies': anomalies_binary.sum(),
        'anomaly_rate': anomalies_binary.mean() * 100,
        'n_estimators': 200,
        'contamination': 0.05
    }
    
    print(f"Isolation Forest detected {results['num_anomalies']} anomalies ({results['anomaly_rate']:.2f}%)")
    
    return results

def run_statistical_graph_anomaly(detector):
    """
    Run statistical anomaly detection enhanced with graph structure
    """
    node_features = detector.graph_data.x.numpy()
    edge_index = detector.graph_data.edge_index.numpy()
    
    # Calculate local anomaly scores based on neighbor comparison
    anomaly_scores = []
    local_densities = []
    neighbor_counts = []
    
    print("Calculating local statistical anomalies...")
    for i in range(len(node_features)):
        # Find neighbors
        neighbors = edge_index[1][edge_index[0] == i]
        neighbor_counts.append(len(neighbors))
        
        if len(neighbors) > 0:
            # Compare with neighbors
            neighbor_features = node_features[neighbors]
            node_feature = node_features[i]
            
            # Calculate multiple deviation measures
            euclidean_devs = np.linalg.norm(neighbor_features - node_feature, axis=1)
            manhattan_devs = np.sum(np.abs(neighbor_features - node_feature), axis=1)
            
            # Local density and isolation measures
            local_density = 1.0 / (1.0 + np.mean(euclidean_devs))
            local_densities.append(local_density)
            
            # Anomaly score combines multiple measures
            local_anomaly_score = (np.mean(euclidean_devs) * 0.5 + 
                                 np.mean(manhattan_devs) * 0.3 + 
                                 (1.0 - local_density) * 0.2)
        else:
            # No neighbors - use global statistics
            global_mean = np.mean(node_features, axis=0)
            local_anomaly_score = np.linalg.norm(node_features[i] - global_mean)
            local_densities.append(0.0)
        
        anomaly_scores.append(local_anomaly_score)
    
    anomaly_scores = np.array(anomaly_scores)
    
    # Store statistical model parameters
    detector.models['statistical_graph'] = {
        'global_mean': np.mean(node_features, axis=0),
        'global_std': np.std(node_features, axis=0),
        'local_density_stats': {
            'mean': np.mean(local_densities),
            'std': np.std(local_densities)
        },
        'neighbor_count_stats': {
            'mean': np.mean(neighbor_counts),
            'std': np.std(neighbor_counts)
        }
    }
    
    # Calculate threshold (95th percentile)
    threshold = np.percentile(anomaly_scores, 95)
    anomalies = anomaly_scores > threshold
    
    results = {
        'anomalies': anomalies,
        'anomaly_scores': anomaly_scores,
        'threshold': threshold,
        'num_anomalies': anomalies.sum(),
        'anomaly_rate': anomalies.mean() * 100,
        'local_densities': local_densities,
        'neighbor_counts': neighbor_counts
    }
    
    print(f"Statistical Graph method detected {results['num_anomalies']} anomalies ({results['anomaly_rate']:.2f}%)")
    
    return results

def generate_comprehensive_anomaly_report(detector, methods_results):
    """
    Generate a comprehensive anomaly detection report
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ANOMALY DETECTION REPORT")
    print("="*80)
    
    # Dataset Summary
    print(f"\n📊 DATASET SUMMARY")
    print(f"{'─'*50}")
    print(f"Total Records: {len(detector.processed_data):,}")
    print(f"Features Used: {len(detector.feature_columns)}")
    print(f"Date Range: {detector.processed_data['S_DATE'].min()} to {detector.processed_data['S_DATE'].max()}")
    print(f"Unique Customers: {detector.processed_data['CUSTOMER_ID'].nunique():,}")
    print(f"Graph Structure: {detector.graph_data.x.shape[0]:,} nodes, {detector.graph_data.edge_index.shape[1]:,} edges")
    
    # Method Performance Comparison
    print(f"\n🔍 METHOD PERFORMANCE COMPARISON")
    print(f"{'─'*80}")
    print(f"{'Method':<25} {'Anomalies':<12} {'Rate %':<10} {'Threshold':<15} {'Score Range':<20}")
    print(f"{'─'*80}")
    
    for method, results in methods_results.items():
        scores = results['anomaly_scores']
        threshold = results.get('threshold', np.percentile(scores, 95))
        print(f"{method:<25} {results['num_anomalies']:<12} {results['anomaly_rate']:<10.2f} "
              f"{threshold:<15.4f} {np.min(scores):.3f} - {np.max(scores):.3f}")
    
    # Detailed Performance Metrics
    print(f"\n📈 DETAILED PERFORMANCE METRICS")
    print(f"{'─'*50}")
    
    for method, metrics in detector.performance_metrics.items():
        print(f"\n{method}:")
        print(f"  Anomaly Detection Rate: {metrics['anomaly_rate']:.2f}%")
        print(f"  Threshold (95th %ile): {metrics['threshold']:.6f}")
        print(f"  Score Statistics:")
        stats = metrics['score_statistics']
        print(f"    Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Quartiles: Q1={stats['q25']:.4f}, Q2={stats['median']:.4f}, Q3={stats['q75']:.4f}")
    
    # Anomaly Analysis by Features
    print(f"\n🎯 ANOMALY ANALYSIS BY FEATURES")
    print(f"{'─'*50}")
    
    # Combine all anomalies for analysis
    all_anomaly_indices = set()
    for results in methods_results.values():
        anomaly_indices = np.where(results['anomalies'])[0]
        all_anomaly_indices.update(anomaly_indices)
    
    if all_anomaly_indices:
        anomaly_data = detector.processed_data.iloc[list(all_anomaly_indices)]
        normal_data = detector.processed_data.drop(list(all_anomaly_indices))
        
        print(f"\nFeature Statistics Comparison (Anomalies vs Normal):")
        print(f"{'Feature':<25} {'Anomaly Mean':<15} {'Normal Mean':<15} {'Difference':<15}")
        print(f"{'─'*70}")
        
        key_features = ['total_upload', 'total_download', 'total_duration', 
                       'upload_download_ratio', 'TOT_SESSIONS', 'zero_kb_ratio']
        
        for feature in key_features:
            if feature in anomaly_data.columns:
                anomaly_mean = anomaly_data[feature].mean()
                normal_mean = normal_data[feature].mean()
                difference = anomaly_mean - normal_mean
                print(f"{feature:<25} {anomaly_mean:<15.2f} {normal_mean:<15.2f} {difference:<15.2f}")
    
    # Consensus Analysis
    print(f"\n🤝 CONSENSUS ANOMALY ANALYSIS")
    print(f"{'─'*50}")
    
    # Count how many methods detected each record as anomaly
    anomaly_counts = np.zeros(len(detector.processed_data))
    method_names = list(methods_results.keys())
    
    for method, results in methods_results.items():
        anomaly_counts += results['anomalies'].astype(int)
    
    for consensus_level in range(1, len(method_names) + 1):
        consensus_anomalies = anomaly_counts >= consensus_level
        count = consensus_anomalies.sum()
        percentage = (count / len(detector.processed_data)) * 100
        print(f"Detected by ≥{consensus_level} method(s): {count:,} records ({percentage:.2f}%)")
    
    # High-confidence anomalies (detected by majority of methods)
    majority_threshold = len(method_names) // 2 + 1
    high_confidence_anomalies = anomaly_counts >= majority_threshold
    
    print(f"\n🚨 HIGH-CONFIDENCE ANOMALIES (≥{majority_threshold} methods)")
    print(f"{'─'*60}")
    print(f"Count: {high_confidence_anomalies.sum():,}")
    
    if high_confidence_anomalies.sum() > 0:
        high_conf_data = detector.processed_data.iloc[high_confidence_anomalies]
        print(f"\nTop 10 High-Confidence Anomalies:")
        display_cols = ['MSISDN', 'CUSTOMER_ID', 'total_upload', 'total_download', 'TOT_SESSIONS']
        print(high_conf_data[display_cols].head(10).to_string(index=False))
    
    # Customer Analysis
    print(f"\n👥 CUSTOMER-BASED ANOMALY ANALYSIS")
    print(f"{'─'*50}")
    
    if all_anomaly_indices:
        anomaly_customers = detector.processed_data.iloc[list(all_anomaly_indices)]['CUSTOMER_ID'].value_counts()
        total_customers_with_anomalies = len(anomaly_customers)
        print(f"Customers with anomalies: {total_customers_with_anomalies:,}")
        print(f"Customers with multiple anomalous records: {(anomaly_customers > 1).sum():,}")
        
        if len(anomaly_customers) > 0:
            print(f"\nTop 5 customers by anomaly count:")
            for customer, count in anomaly_customers.head().items():
                print(f"  Customer {customer}: {count} anomalous records")
    
    # Time-based Analysis
    print(f"\n📅 TEMPORAL ANOMALY ANALYSIS")
    print(f"{'─'*50}")
    
    if all_anomaly_indices:
        anomaly_dates = detector.processed_data.iloc[list(all_anomaly_indices)]
        anomaly_dates['date'] = pd.to_datetime(anomaly_dates['S_DATE']).dt.date
        date_counts = anomaly_dates['date'].value_counts().head(10)
        
        print(f"Dates with most anomalies:")
        for date, count in date_counts.items():
            print(f"  {date}: {count} anomalies")
    
    # Technology Usage Anomalies
    print(f"\n📡 TECHNOLOGY USAGE ANOMALY PATTERNS")
    print(f"{'─'*50}")
    
    if all_anomaly_indices:
        tech_dist = detector.processed_data.iloc[list(all_anomaly_indices)]['dominant_tech'].value_counts()
        print(f"Anomalies by dominant technology:")
        for tech, count in tech_dist.items():
            percentage = (count / len(all_anomaly_indices)) * 100
            print(f"  {tech}G: {count} anomalies ({percentage:.1f}%)")
    
    # Store comprehensive report
    detector.anomaly_report = {
        'dataset_summary': {
            'total_records': len(detector.processed_data),
            'features_count': len(detector.feature_columns),
            'unique_customers': detector.processed_data['CUSTOMER_ID'].nunique(),
            'graph_nodes': detector.graph_data.x.shape[0],
            'graph_edges': detector.graph_data.edge_index.shape[1]
        },
        'method_performance': {method: {
            'num_anomalies': results['num_anomalies'],
            'anomaly_rate': results['anomaly_rate'],
            'threshold': results.get('threshold', np.percentile(results['anomaly_scores'], 95))
        } for method, results in methods_results.items()},
        'consensus_analysis': {
            f'consensus_{i}': int((anomaly_counts >= i).sum()) 
            for i in range(1, len(method_names) + 1)
        },
        'high_confidence_anomalies': int(high_confidence_anomalies.sum()),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n✅ Report generated successfully!")
    return detector.anomaly_report

def visualize_comprehensive_results(detector, methods_results):
    """
    Create comprehensive visualization of anomaly detection results
    """
    fig = plt.figure(figsize=(20, 16))
    
    # PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(detector.graph_data.x.numpy())
    
    methods = list(methods_results.keys())
    
    # 1. Anomaly Detection Results by Method
    for i, method in enumerate(methods):
        plt.subplot(3, 4, i + 1)
        results = methods_results[method]
        
        normal_mask = ~results['anomalies']
        anomaly_mask = results['anomalies']
        
        plt.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Normal', s=20)
        plt.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                   c='red', alpha=0.8, label='Anomaly', s=30)
        
        plt.title(f'{method}\nAnomalies: {results["num_anomalies"]} ({results["anomaly_rate"]:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Anomaly Score Distributions
    for i, method in enumerate(methods):
        plt.subplot(3, 4, i + 4)
        results = methods_results[method]
        
        plt.hist(results['anomaly_scores'], bins=50, alpha=0.7, color='skyblue', density=True)
        plt.axvline(results.get('threshold', np.percentile(results['anomaly_scores'], 95)), 
                   color='red', linestyle='--', linewidth=2, label='Threshold')
        plt.title(f'{method} - Score Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Performance Comparison
    plt.subplot(3, 4, 9)
    method_names = list(methods_results.keys())
    anomaly_rates = [results['anomaly_rate'] for results in methods_results.values()]
    
    bars = plt.bar(method_names, anomaly_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Anomaly Detection Rates by Method')
    plt.ylabel('Anomaly Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, anomaly_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. Feature Importance (for anomalies vs normal)
    plt.subplot(3, 4, 10)
    
    # Combine all anomalies
    all_anomaly_indices = set()
    for results in methods_results.values():
        anomaly_indices = np.where(results['anomalies'])[0]
        all_anomaly_indices.update(anomaly_indices)
    
    if all_anomaly_indices:
        anomaly_data = detector.processed_data.iloc[list(all_anomaly_indices)]
        normal_data = detector.processed_data.drop(list(all_anomaly_indices))
        
        key_features = ['total_upload', 'total_download', 'total_duration', 'TOT_SESSIONS']
        feature_differences = []
        
        for feature in key_features:
            if feature in anomaly_data.columns:
                anomaly_mean = anomaly_data[feature].mean()
                normal_mean = normal_data[feature].mean()
                # Normalize difference
                if normal_mean != 0:
                    diff = (anomaly_mean - normal_mean) / normal_mean * 100
                else:
                    diff = 0
                feature_differences.append(diff)
        
        bars = plt.bar(key_features, feature_differences, 
                      color=['red' if x > 0 else 'blue' for x in feature_differences])
        plt.title('Feature Differences\n(Anomaly vs Normal %)')
        plt.ylabel('Percentage Difference')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 5. Consensus Analysis
    plt.subplot(3, 4, 11)
    
    anomaly_counts = np.zeros(len(detector.processed_data))
    for results in methods_results.values():
        anomaly_counts += results['anomalies'].astype(int)
    
    consensus_levels = list(range(1, len(methods) + 1))
    consensus_counts = [(anomaly_counts >= level).sum() for level in consensus_levels]
    
    plt.bar(consensus_levels, consensus_counts, color='orange', alpha=0.7)
    plt.title('Consensus Anomaly Detection')
    plt.xlabel('Number of Methods Agreeing')
    plt.ylabel('Number of Records')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Training Loss (for GNN AutoEncoder)
    plt.subplot(3, 4, 12)
    
    if 'GNN_AutoEncoder' in methods_results and 'training_losses' in methods_results['GNN_AutoEncoder']:
        losses = methods_results['GNN_AutoEncoder']['training_losses']
        plt.plot(losses, color='purple', linewidth=2)
        plt.title('GNN AutoEncoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'anomaly_detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_prediction_report(detector, results):
    """
    Generate report for predictions on new data
    """
    print(f"\n📋 PREDICTION REPORT ON NEW DATA")
    print(f"{'='*60}")
    print(f"Dataset: {detector.data_path}")
    print(f"Records processed: {len(detector.processed_data):,}")
    print(f"Prediction timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🔍 ANOMALY DETECTION RESULTS")
    print(f"{'─'*50}")
    
    for method, method_results in results.items():
        print(f"{method}:")
        print(f"  Anomalies detected: {method_results['num_anomalies']:,}")
        print(f"  Anomaly rate: {method_results['anomaly_rate']:.2f}%")
        print(f"  Threshold used: {method_results['threshold']:.6f}")
    
    # High-risk records
    all_anomaly_indices = set()
    for method_results in results.values():
        anomaly_indices = np.where(method_results['anomalies'])[0]
        all_anomaly_indices.update(anomaly_indices)
    
    if all_anomaly_indices:
        print(f"\n🚨 HIGH-RISK RECORDS")
        print(f"{'─'*30}")
        high_risk_data = detector.processed_data.iloc[list(all_anomaly_indices)]
        print(f"Records flagged: {len(high_risk_data):,}")
        
        # Show top anomalies
        print(f"\nTop 5 highest-risk records:")
        display_cols = ['MSISDN', 'CUSTOMER_ID', 'total_upload', 'total_download']
        print(high_risk_data[display_cols].head().to_string(index=False))
    
    return {
        'total_records': len(detector.processed_data),
        'methods_used': list(results.keys()),
        'total_anomalies': len(all_anomaly_indices),
        'anomaly_rate': len(all_anomaly_indices) / len(detector.processed_data) * 100
    }

# Prediction methods for new data
def _predict_gnn_autoencoder(detector):
    """Predict anomalies using trained GNN AutoEncoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_info = detector.models['gnn_autoencoder']
    model = model_info['model'].to(device)
    
    data = detector.graph_data.to(device)
    model.eval()
    
    with torch.no_grad():
        x_reconstructed, z = model(data.x, data.edge_index)
        reconstruction_errors = F.mse_loss(x_reconstructed, data.x, reduction='none').mean(dim=1)
        
        # Use same threshold logic as training
        threshold = torch.quantile(reconstruction_errors, 0.95)
        anomalies = (reconstruction_errors > threshold).cpu().numpy()
        anomaly_scores = reconstruction_errors.cpu().numpy()
    
    return {
        'anomalies': anomalies,
        'anomaly_scores': anomaly_scores,
        'threshold': threshold.item(),
        'num_anomalies': anomalies.sum(),
        'anomaly_rate': anomalies.mean() * 100
    }

def _predict_isolation_forest(detector):
    """Predict anomalies using trained Isolation Forest"""
    node_features = detector.graph_data.x.numpy()
    G = to_networkx(detector.graph_data, to_undirected=True)
    
    # Calculate same graph features as training
    degree_centrality = list(nx.degree_centrality(G).values())
    betweenness_centrality = list(nx.betweenness_centrality(G, k=min(100, len(G))).values())
    clustering_coeff = list(nx.clustering(G).values())
    
    try:
        pagerank = list(nx.pagerank(G, max_iter=100).values())
        eigenvector_centrality = list(nx.eigenvector_centrality(G, max_iter=100).values())
    except:
        pagerank = [0] * len(G)
        eigenvector_centrality = [0] * len(G)
    
    graph_features = np.column_stack([
        node_features, degree_centrality, betweenness_centrality,
        clustering_coeff, pagerank, eigenvector_centrality
    ])
    
    model = detector.models['isolation_forest']['model']
    anomalies = model.predict(graph_features)
    anomaly_scores = model.score_samples(graph_features)
    
    anomalies_binary = (anomalies == -1)
    
    return {
        'anomalies': anomalies_binary,
        'anomaly_scores': -anomaly_scores,
        'num_anomalies': anomalies_binary.sum(),
        'anomaly_rate': anomalies_binary.mean() * 100,
        'threshold': np.percentile(-anomaly_scores, 95)
    }

def _predict_statistical_graph(detector):
    """Predict anomalies using statistical graph method"""
    node_features = detector.graph_data.x.numpy()
    edge_index = detector.graph_data.edge_index.numpy()
    
    model_params = detector.models['statistical_graph']
    
    anomaly_scores = []
    for i in range(len(node_features)):
        neighbors = edge_index[1][edge_index[0] == i]
        
        if len(neighbors) > 0:
            neighbor_features = node_features[neighbors]
            node_feature = node_features[i]
            
            euclidean_devs = np.linalg.norm(neighbor_features - node_feature, axis=1)
            manhattan_devs = np.sum(np.abs(neighbor_features - node_feature), axis=1)
            
            local_density = 1.0 / (1.0 + np.mean(euclidean_devs))
            local_anomaly_score = (np.mean(euclidean_devs) * 0.5 + 
                                 np.mean(manhattan_devs) * 0.3 + 
                                 (1.0 - local_density) * 0.2)
        else:
            global_mean = model_params['global_mean']
            local_anomaly_score = np.linalg.norm(node_features[i] - global_mean)
        
        anomaly_scores.append(local_anomaly_score)
    
    anomaly_scores = np.array(anomaly_scores)
    threshold = np.percentile(anomaly_scores, 95)
    anomalies = anomaly_scores > threshold
    
    return {
        'anomalies': anomalies,
        'anomaly_scores': anomaly_scores,
        'threshold': threshold,
        'num_anomalies': anomalies.sum(),
        'anomaly_rate': anomalies.mean() * 100
    }

# Add prediction methods to detector class
TelecomGNNAnomalyDetector._predict_gnn_autoencoder = _predict_gnn_autoencoder
TelecomGNNAnomalyDetector._predict_isolation_forest = _predict_isolation_forest  
TelecomGNNAnomalyDetector._predict_statistical_graph = _predict_statistical_graph

# Main execution with enhanced functionality
if __name__ == "__main__":
    print("="*100)
    print("🚀 ADVANCED GNN-based ANOMALY DETECTION for TELECOMMUNICATIONS DATA")
    print("="*100)
    
    # Initialize detector
    detector = TelecomGNNAnomalyDetector('D:/Work/data.csv')
    
    # Load and preprocess data
    detector.load_and_preprocess_data()
    
    # Create graph structure
    detector.create_graph_structure(similarity_threshold=0.7)
    
    # Run anomaly detection methods
    print("\n🔄 Running anomaly detection methods...")
    results = run_anomaly_detection_methods(detector)
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive anomaly report...")
    anomaly_report = generate_comprehensive_anomaly_report(detector, results)
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    visualize_comprehensive_results(detector, results)
    
    # Save trained model
    print("\n💾 Saving trained models...")
    model_path = detector.save_model()
    print(f"✅ Models saved to: {model_path}.pkl")
    
    print(f"\n📋 FINAL SUMMARY:")
    print(f"  - Total records analyzed: {len(detector.processed_data):,}")
    print(f"  - Methods used: {len(results)}")
    print(f"  - High-confidence anomalies: {anomaly_report['high_confidence_anomalies']:,}")
    print(f"  - Model ready for production use!")
    
    print(f"\n💡 To use this model on new data:")
    print(f"  1. Initialize: detector = TelecomGNNAnomalyDetector()")
    print(f"  2. Load model: detector.load_model('{model_path}.pkl')")
    print(f"  3. Predict: results, report = detector.predict_new_data('new_data.csv')")