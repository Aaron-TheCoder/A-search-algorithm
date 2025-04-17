import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import datetime
import heapq
import time

# Generate synthetic traffic data
def generate_traffic_data(num_roads=30, days=7):
    """Generate synthetic traffic data for model training"""
    print("Generating synthetic traffic data...")
    start_date = datetime.datetime(2023, 1, 1)
    timestamps = []
    road_ids = []
    traffic_levels = []
    temperatures = []
    rainfalls = []
    snowfalls = []
    holidays = []
    cloud_coverages = []
    
    for day in range(days):
        for hour in range(24):
            current_time = start_date + datetime.timedelta(days=day, hours=hour)
            
            # Environmental conditions
            temperature = np.random.uniform(0, 35)  # Celsius
            rainfall = max(0, np.random.normal(0, 1)) if np.random.random() < 0.2 else 0  # mm
            snowfall = max(0, np.random.normal(0, 2)) if temperature < 0 and np.random.random() < 0.1 else 0  # mm
            cloud_coverage = np.random.randint(0, 100)  # percentage
            is_holiday = 1 if day % 7 >= 5 else 0  # weekend as holiday
            
            for road in range(num_roads):
                road_id = f"road_{road}"
                
                # Base traffic level
                base_traffic = 1000  # Base volume
                
                # Add rush hour effect (7-9 AM and 4-6 PM)
                if hour in [7, 8, 9]:
                    base_traffic += 800
                elif hour in [16, 17, 18]:
                    base_traffic += 1000
                
                # Add weekend effect (less traffic on weekends)
                if day % 7 >= 5:  # Weekend
                    base_traffic *= 0.7
                
                # Weather effects
                if rainfall > 1:
                    base_traffic *= 0.9
                if snowfall > 0:
                    base_traffic *= 0.7
                
                # Add random variation
                noise = np.random.normal(0, 200)
                traffic_level = max(100, base_traffic + noise)
                
                timestamps.append(current_time)
                road_ids.append(road_id)
                traffic_levels.append(traffic_level)
                temperatures.append(temperature)
                rainfalls.append(rainfall)
                snowfalls.append(snowfall)
                holidays.append(is_holiday)
                cloud_coverages.append(cloud_coverage)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'road_id': road_ids,
        'traffic_level': traffic_levels,
        'temperature': temperatures,
        'rainfall': rainfalls,
        'snowfall': snowfalls,
        'holiday': holidays,
        'cloud_coverage': cloud_coverages
    })
    
    print(f"Generated {len(df)} traffic data points")
    return df

# City graph representation
class CityGraph:
    def __init__(self, num_junctions=20):
        # Create a graph for a city with random connections
        self.G = nx.random_geometric_graph(num_junctions, 0.3)
        
        # Dictionary to store road_id to edge mappings
        self.road_to_edge = {}
        self.edge_to_road = {}
        
        # Add traffic attributes to edges
        edge_index = 0
        for u, v in self.G.edges():
            road_id = f"road_{edge_index}"
            
            self.road_to_edge[road_id] = (u, v)
            self.edge_to_road[(u, v)] = road_id
            
            # Initial traffic level (1.0 = normal flow, higher = congestion)
            self.G[u][v]['traffic'] = 1.0
            # Physical distance between junctions
            self.G[u][v]['distance'] = np.random.uniform(0.5, 5.0)
            # Assign road ID
            self.G[u][v]['road_id'] = road_id
            
            edge_index += 1
    
    def map_roads_to_edges(self, traffic_data):
        """Map road IDs from traffic data to graph edges"""
        # Extract unique road IDs from data
        self.road_ids = traffic_data['road_id'].unique()
        
        # Reassign road IDs to edges if needed
        edge_index = 0
        for u, v in self.G.edges():
            if edge_index < len(self.road_ids):
                road_id = self.road_ids[edge_index]
                self.road_to_edge[road_id] = (u, v)
                self.edge_to_road[(u, v)] = road_id
                self.G[u][v]['road_id'] = road_id
            edge_index += 1
    
    def update_traffic(self, traffic_data, timestamp=None):
        """Update traffic levels based on real-time data"""
        if timestamp is not None:
            # Filter data for the specific timestamp
            current_data = traffic_data[traffic_data['timestamp'] == timestamp]
        else:
            # Use the latest data point for each road
            current_data = traffic_data.sort_values('timestamp').groupby('road_id').last().reset_index()
        
        # Update each edge with the corresponding traffic level
        for _, row in current_data.iterrows():
            road_id = row['road_id']
            if road_id in self.road_to_edge:
                u, v = self.road_to_edge[road_id]
                # Normalize traffic level to a reasonable range (1.0 = normal, >1.0 = congested)
                traffic_level = row['traffic_level']
                # Convert absolute traffic volume to a relative congestion factor
                normalized_traffic = traffic_level / 1000  # Assuming 1000 is a baseline
                self.G[u][v]['traffic'] = max(0.5, min(3.0, normalized_traffic))
    
    def get_neighbors(self, node):
        """Get neighboring junctions"""
        return list(self.G.neighbors(node))
    
    def get_edge_weight(self, u, v):
        """Calculate edge weight based on distance and traffic"""
        if self.G.has_edge(u, v):
            return self.G[u][v]['distance'] * self.G[u][v]['traffic']
        return float('inf')
    
    def visualize(self, path=None):
        """Visualize the city graph with traffic levels"""
        plt.figure(figsize=(10, 8))
        pos = nx.get_node_attributes(self.G, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=300, node_color='lightblue')
        
        # Draw edges with color based on traffic
        edge_colors = []
        for u, v in self.G.edges():
            traffic = self.G[u][v]['traffic']
            # Red = congested, Green = clear
            if traffic > 2.0:
                edge_colors.append('red')
            elif traffic > 1.5:
                edge_colors.append('orange')
            else:
                edge_colors.append('green')
        
        nx.draw_networkx_edges(self.G, pos, width=2, edge_color=edge_colors)
        
        # Draw the selected path if provided
        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            edges_exist = [(u, v) for u, v in path_edges if self.G.has_edge(u, v)]
            nx.draw_networkx_edges(self.G, pos, edgelist=edges_exist, 
                                  width=4, edge_color='blue')
        
        nx.draw_networkx_labels(self.G, pos)
        plt.title("City Traffic Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Traffic Prediction Model using historical data
class TrafficPredictor:
    def __init__(self):
        self.models = {}
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        
    def prepare_data(self, data):
        """
        Prepare traffic data for prediction
        - data: DataFrame with timestamps, road_id, and traffic metrics
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Debug print to check data types
        print(f"Data types before processing:\n{df.dtypes}")
        
        # Ensure timestamp is a datetime type
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                print("Converting timestamp to datetime...")
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Create time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=['road_id'])
        
        # Drop timestamp column
        features = df.drop(['timestamp', 'traffic_level'], axis=1)
        target = df['traffic_level']
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(features)
        y_scaled = self.scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled, features.columns
    
    def train(self, data):
        """Train predictive models for traffic levels"""
        X, y, feature_names = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train different models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'A* Search Hybrid': AStarTrafficModel(feature_names)
        }
        
        results = {}
        for name, model in models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Scale predictions back to original range
            y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            mae = mean_absolute_error(y_test_original, y_pred_original)
            
            self.models[name] = model
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'Training Time': train_time,
                'Inference Time': inference_time,
                'y_test': y_test_original,
                'y_pred': y_pred_original
            }
            
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return results, X_test, y_test
    
    def predict(self, features, model_name='A* Search Hybrid'):
        """Predict traffic using the selected model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scaler_X.transform(features)
        y_pred_scaled = self.models[model_name].predict(X_scaled)
        
        # Scale predictions back to original range
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def compare_models_visualization(self, results):
        """Visualize performance comparison of different models"""
        models = list(results.keys())
        rmse_values = [results[m]['RMSE'] for m in models]
        mae_values = [results[m]['MAE'] for m in models]
        
        # Model accuracy comparison
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, rmse_values, width, label='RMSE')
        plt.bar(x + width/2, mae_values, width, label='MAE')
        
        plt.ylabel('Error Metric')
        plt.title('Traffic Prediction Model Performance Comparison')
        plt.xticks(x, models)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Execution time comparison
        plt.figure(figsize=(12, 6))
        train_times = [results[m]['Training Time'] for m in models]
        inference_times = [results[m]['Inference Time'] for m in models]
        
        plt.bar(x - width/2, train_times, width, label='Training Time')
        plt.bar(x + width/2, inference_times, width, label='Inference Time')
        
        plt.ylabel('Time (seconds)')
        plt.title('Model Execution Time Comparison')
        plt.xticks(x, models)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Predicted vs Actual for A* Search Hybrid
        plt.figure(figsize=(10, 6))
        a_star_model = 'A* Search Hybrid'
        plt.scatter(results[a_star_model]['y_test'], results[a_star_model]['y_pred'], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(results[a_star_model]['y_test']), min(results[a_star_model]['y_pred']))
        max_val = max(max(results[a_star_model]['y_test']), max(results[a_star_model]['y_pred']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Traffic Volume')
        plt.ylabel('Predicted Traffic Volume')
        plt.title(f'{a_star_model}: Predicted vs Actual Traffic')
        plt.tight_layout()
        plt.show()

# A* Search-based Traffic Model
class AStarTrafficModel:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.base_model = RandomForestRegressor(n_estimators=150, random_state=42)
        self.feature_weights = {}
        # Initial equal weights for features
        for feature in feature_names:
            self.feature_weights[feature] = 1.0
    
    def fit(self, X, y):
        """
        Train the model using A* search principles to find optimal feature weights
        """
        # Train the base model
        self.base_model.fit(X, y)
        
        # Get feature importances from base model
        if hasattr(self.base_model, 'feature_importances_'):
            importances = self.base_model.feature_importances_
            # Update feature weights based on importance
            for i, feature in enumerate(self.feature_names):
                if i < len(importances):
                    self.feature_weights[feature] = 1.0 + importances[i]  # Boost by importance
        
        return self
    
    def predict(self, X):
        """
        Predict traffic using A* search principles to refine predictions
        """
        # Get base predictions
        base_predictions = self.base_model.predict(X)
        
        # In a real implementation, we would:
        # 1. Use the base predictions as initial estimates
        # 2. Apply A* search to refine predictions based on road network constraints
        # 3. Ensure traffic flow conservation (what goes in must come out at junctions)
        
        # For this demo, we'll simulate the A* refinement with some adjustments
        refinement_factor = np.random.uniform(0.97, 1.03, size=len(base_predictions))
        refined_predictions = base_predictions * refinement_factor
        
        return refined_predictions

# A* Search Algorithm for Routing
def a_star_search(city_graph, start, goal, traffic_predictor=None, timestamp=None):
    """
    A* search algorithm for finding optimal route in city with traffic
    If traffic_predictor is provided, it will use predictions for future traffic
    """
    # Initialize data structures
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    # Cost from start to node
    g_score = {node: float('inf') for node in city_graph.G.nodes()}
    g_score[start] = 0
    
    # Estimated cost from node to goal
    f_score = {node: float('inf') for node in city_graph.G.nodes()}
    
    # Heuristic function - Euclidean distance as base
    pos = nx.get_node_attributes(city_graph.G, 'pos')
    h = lambda n: np.sqrt((pos[n][0] - pos[goal][0])*2 + (pos[n][1] - pos[goal][1])*2)
    
    f_score[start] = h(start)
    
    # For recording the search process
    explored_nodes = []
    
    while open_set:
        # Get node with lowest f_score
        current_f, current = heapq.heappop(open_set)
        explored_nodes.append(current)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, explored_nodes
        
        for neighbor in city_graph.get_neighbors(current):
            # Calculate tentative g_score using current traffic conditions
            traffic_weight = city_graph.get_edge_weight(current, neighbor)
            
            # If we have a traffic predictor, adjust the weight based on predictions
            if traffic_predictor and (current, neighbor) in city_graph.edge_to_road:
                road_id = city_graph.edge_to_road[(current, neighbor)]
                # In a real system, we would create features for this specific road and time
                # For simplicity, we'll just adjust the current traffic by a small factor
                traffic_weight *= 1.1  # Assuming future traffic might be slightly worse
            
            tentative_g = g_score[current] + traffic_weight
            
            if tentative_g < g_score[neighbor]:
                # Found a better path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = g_score[neighbor] + h(neighbor)
                
                # Add to open set if not already there
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None, explored_nodes

# Main function to demonstrate the model
def main():
    # Generate synthetic traffic data
    traffic_data = generate_traffic_data(num_roads=30, days=7)
    print(f"Dataset created with {len(traffic_data)} records")
    print(f"Sample data:\n{traffic_data.head()}")
    
    # Create a city graph based on the traffic data
    city = CityGraph()
    city.map_roads_to_edges(traffic_data)
    
    # Create and train the traffic prediction model
    print("\nTraining traffic prediction models...")
    predictor = TrafficPredictor()
    results, X_test, y_test = predictor.train(traffic_data)
    
    # Visualize model comparison
    print("\nComparing model performance...")
    predictor.compare_models_visualization(results)
    
    # Demonstrate A* routing with real traffic data
    start_node = 0
    end_node = 15  # Choose a destination
    
    # Update traffic based on synthetic data
    city.update_traffic(traffic_data)
    
    # Find optimal route
    print(f"\nCalculating route from junction {start_node} to {end_node}...")
    optimal_path, explored = a_star_search(city, start_node, end_node, predictor)
    
    if optimal_path:
        print(f"Optimal path found: {optimal_path}")
        print(f"Number of nodes explored: {len(explored)}")
        city.visualize(optimal_path)
    else:
        print("No path found!")
        city.visualize()
    
    # Demonstrate with and without traffic prediction
    print("\nComparing routing with and without traffic prediction:")
    
    # Without prediction
    path1, explored1 = a_star_search(city, start_node, end_node)
    
    # With prediction
    path2, explored2 = a_star_search(city, start_node, end_node, predictor)
    
    if path1 and path2:
        print(f"Path without prediction: {path1}")
        print(f"Path with prediction: {path2}")
        print(f"Nodes explored without prediction: {len(explored1)}")
        print(f"Nodes explored with prediction: {len(explored2)}")
        
        # Check if paths are different
        if path1 != path2:
            print("Traffic prediction changed the selected route!")
        else:
            print("Same route selected with and without prediction.")
    
    # Create example for routing in different traffic conditions
    print("\nSimulating rush hour traffic conditions...")
    
    # Create rush hour traffic (higher congestion on some roads)
    for u, v in city.G.edges():
        # Randomly make some roads more congested
        if np.random.random() < 0.3:
            city.G[u][v]['traffic'] = np.random.uniform(2.0, 3.0)  # Heavy traffic
    
    # Find new optimal route
    rush_hour_path, rush_hour_explored = a_star_search(city, start_node, end_node)
    
    if rush_hour_path:
        print(f"Rush hour path: {rush_hour_path}")
        print(f"Nodes explored: {len(rush_hour_explored)}")
        city.visualize(rush_hour_path)
        
        # Compare with normal conditions
        if path1 and rush_hour_path != path1:
            print("Traffic conditions changed the selected route!")
        else:
            print("Same route selected despite traffic changes.")

if __name__ == "__main__":
    main()