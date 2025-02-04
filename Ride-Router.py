import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
from geopy.distance import geodesic
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy.random as rnd
import googlemaps
import polyline
from folium.plugins import GroupedLayerControl
from dotenv import find_dotenv,load_dotenv
import os
load_dotenv()

st.set_page_config(layout="wide")

class SOMCluster:
    def __init__(self, input_len, grid_size=3, sigma=1.0, learning_rate=0.5):
        self.grid_size = grid_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.input_len = input_len
        self._init_weights()
        
    def _init_weights(self):
        self.weights = rnd.rand(self.grid_size, self.grid_size, self.input_len)
        
    def _neighborhood(self, c, sigma):
        d = 2*sigma*sigma
        ax = np.arange(self.grid_size)
        xx, yy = np.meshgrid(ax, ax)
        return np.exp(-((xx-c[0])**2 + (yy-c[1])**2) / d)

    def find_winner(self, x):
        diff = self.weights - x
        dist = np.sum(diff**2, axis=-1)
        return np.unravel_index(np.argmin(dist), dist.shape)
    
    def train(self, data, epochs=2000):
        for epoch in range(epochs):
            sigma = self.sigma * (1 - epoch/epochs)
            lr = self.learning_rate * (1 - epoch/epochs)
            
            for x in data:
                winner = self.find_winner(x)
                g = self._neighborhood(winner, sigma)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        self.weights[i,j] += lr * g[i,j] * (x - self.weights[i,j])

    def get_cluster(self, x):
        winner = self.find_winner(x)
        return winner[0] * self.grid_size + winner[1]

class StaffTransportOptimizer:
    def __init__(self, google_maps_key):
        self.office_location = {
            'lat': 5.582636441579255,
            'lon': -0.143551646497661
        }
        self.MAX_PASSENGERS = 4
        self.MIN_PASSENGERS = 3
        self.COST_PER_KM = 2.5
        self.scaler = MinMaxScaler()
        self.gmaps = googlemaps.Client(key=google_maps_key)
    def _assign_remaining_staff(self, remaining_staff, routes):
        """Assign remaining staff to existing routes"""
        if not isinstance(remaining_staff, pd.DataFrame):
            return
            
        for idx, row in remaining_staff.iterrows():
            best_route = None
            min_detour = float('inf')
            staff_dict = row.to_dict()
            
            for route_name, route_group in routes.items():
                if len(route_group) < self.MAX_PASSENGERS:
                    # Calculate current route distance
                    current_distance = self.calculate_route_metrics(route_group)[0]
                    
                    # Calculate new route distance with added staff member
                    test_route = route_group.copy()
                    test_route.append(staff_dict)
                    new_distance = self.calculate_route_metrics(test_route)[0]
                    
                    # Calculate detour distance
                    detour = new_distance - current_distance
                    
                    if detour < min_detour:
                        min_detour = detour
                        best_route = route_name
            
            if best_route:
                routes[best_route].append(staff_dict)
    def validate_staff_data(self, df):
        """Validate and clean staff location data"""
        try:
            # Check required columns
            required_columns = ['staff_id', 'name', 'latitude', 'longitude', 'address']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Create a copy to avoid modifying the original dataframe
            clean_df = df.copy()
            
            # Convert coordinates to numeric values
            clean_df['latitude'] = pd.to_numeric(clean_df['latitude'], errors='coerce')
            clean_df['longitude'] = pd.to_numeric(clean_df['longitude'], errors='coerce')
            
            # Remove rows with invalid coordinates
            invalid_coords = clean_df[clean_df['latitude'].isna() | clean_df['longitude'].isna()]
            if not invalid_coords.empty:
                st.warning(f"Removed {len(invalid_coords)} entries with invalid coordinates")
                clean_df = clean_df.dropna(subset=['latitude', 'longitude'])
            
            # Validate coordinate ranges for Ghana
            valid_lat_range = (4.5, 11.5)  # Ghana's latitude range
            valid_lon_range = (-3.5, 1.5)  # Ghana's longitude range
            
            # Create mask for valid coordinates
            coord_mask = (
                (clean_df['latitude'].between(*valid_lat_range)) &
                (clean_df['longitude'].between(*valid_lon_range))
            )
            
            invalid_range = clean_df[~coord_mask]
            if not invalid_range.empty:
                st.warning(f"Removed {len(invalid_range)} entries with coordinates outside Ghana")
                clean_df = clean_df[coord_mask]
            
            # Remove duplicates based on staff_id
            duplicates = clean_df[clean_df.duplicated(subset=['staff_id'], keep='first')]
            if not duplicates.empty:
                st.warning(f"Removed {len(duplicates)} duplicate staff entries")
                clean_df = clean_df.drop_duplicates(subset=['staff_id'], keep='first')
            
            # Validate minimum required staff
            if len(clean_df) < self.MIN_PASSENGERS:
                raise ValueError(
                    f"Need at least {self.MIN_PASSENGERS} valid staff locations, "
                    f"but only found {len(clean_df)}"
                )
            
            # Validate data types
            clean_df['staff_id'] = clean_df['staff_id'].astype(str)
            clean_df['name'] = clean_df['name'].astype(str)
            clean_df['address'] = clean_df['address'].astype(str)
            
            # Add distance to office column
            clean_df['distance_to_office'] = clean_df.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            st.success(f"Successfully validated {len(clean_df)} staff records")
            return clean_df
            
        except Exception as e:
            st.error(f"Error validating staff data: {str(e)}")
            return None

    def load_sample_data(self):
        """Load sample staff location data for testing"""
        try:
            sample_data = pd.DataFrame({
                'staff_id': range(1, 21),
                'name': [f'Employee {i}' for i in range(1, 21)],
                'latitude': np.random.uniform(5.5526, 5.6126, 20),
                'longitude': np.random.uniform(-0.1735, -0.1135, 20),
                'address': [
                    'Adabraka', 'Osu', 'Cantonments', 'Airport Residential',
                    'East Legon', 'Spintex', 'Tema', 'Teshie', 'Labadi',
                    'Labone', 'Ridge', 'Roman Ridge', 'Dzorwulu', 'Abelemkpe',
                    'North Kaneshie', 'Dansoman', 'Mamprobi', 'Chorkor',
                    'Abeka', 'Achimota'
                ]
            })
            
            # Validate the sample data
            return self.validate_staff_data(sample_data)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            return None
    def create_clusters(self, staff_data, grid_size=3, sigma=1.0, learning_rate=0.5):
        """Create clusters based on staff locations using SOM"""
        if staff_data is None or len(staff_data) == 0:
            return None
        
        try:
            # Calculate distances to office
            staff_data['distance_to_office'] = staff_data.apply(
                lambda row: geodesic(
                    (row['latitude'], row['longitude']),
                    (self.office_location['lat'], self.office_location['lon'])
                ).km,
                axis=1
            )
            
            # Prepare data for clustering
            locations = staff_data[['latitude', 'longitude', 'distance_to_office']].values
            normalized_data = self.scaler.fit_transform(locations)
            
            # Initialize and train SOM
            som = SOMCluster(
                input_len=3,
                grid_size=grid_size,
                sigma=sigma,
                learning_rate=learning_rate
            )
            
            som.train(normalized_data)
            
            # Assign clusters
            staff_data['cluster'] = [som.get_cluster(loc) for loc in normalized_data]
            
            # Handle small clusters
            self._handle_small_clusters(staff_data)
            
            return staff_data
            
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
            return None

    def _handle_small_clusters(self, staff_data):
        """Handle clusters with fewer than minimum required passengers"""
        cluster_sizes = staff_data['cluster'].value_counts()
        small_clusters = cluster_sizes[cluster_sizes < self.MIN_PASSENGERS].index
        
        if len(small_clusters) > 0:
            for small_cluster in small_clusters:
                small_cluster_points = staff_data[staff_data['cluster'] == small_cluster]
                
                for idx, row in small_cluster_points.iterrows():
                    distances = []
                    for cluster_id in staff_data['cluster'].unique():
                        if cluster_id not in small_clusters:
                            cluster_points = staff_data[staff_data['cluster'] == cluster_id]
                            if not cluster_points.empty:
                                avg_dist = cluster_points.apply(
                                    lambda x: geodesic(
                                        (row['latitude'], row['longitude']),
                                        (x['latitude'], x['longitude'])
                                    ).km,
                                    axis=1
                                ).mean()
                                distances.append((cluster_id, avg_dist))
                    
                    if distances:
                        nearest_cluster = min(distances, key=lambda x: x[1])[0]
                        staff_data.at[idx, 'cluster'] = nearest_cluster

    def optimize_routes(self, staff_data):
        """Optimize routes within each cluster"""
        routes = defaultdict(list)
        route_counter = 0
        
        try:
            if 'distance_to_office' not in staff_data.columns:
                staff_data['distance_to_office'] = staff_data.apply(
                    lambda row: geodesic(
                        (row['latitude'], row['longitude']),
                        (self.office_location['lat'], self.office_location['lon'])
                    ).km,
                    axis=1
                )
            
            for cluster_id in staff_data['cluster'].unique():
                cluster_group = staff_data[staff_data['cluster'] == cluster_id].copy()
                
                while len(cluster_group) >= self.MIN_PASSENGERS:
                    current_route = []
                    remaining = cluster_group.copy()
                    
                    # Start with furthest person from office
                    start_person = remaining.nlargest(1, 'distance_to_office').iloc[0]
                    current_route.append(start_person.to_dict())
                    remaining = remaining.drop(start_person.name)
                    
                    while len(current_route) < self.MAX_PASSENGERS and not remaining.empty:
                        last_point = current_route[-1]
                        
                        remaining['temp_distance'] = remaining.apply(
                            lambda row: geodesic(
                                (last_point['latitude'], last_point['longitude']),
                                (row['latitude'], row['longitude'])
                            ).km,
                            axis=1
                        )
                        
                        next_person = remaining.nsmallest(1, 'temp_distance').iloc[0]
                        current_route.append(next_person.to_dict())
                        remaining = remaining.drop(next_person.name)
                        
                        if len(current_route) >= self.MIN_PASSENGERS:
                            break
                    
                    if len(current_route) >= self.MIN_PASSENGERS:
                        route_name = f'Route {route_counter + 1}'
                        routes[route_name] = current_route
                        route_counter += 1
                        assigned_ids = [p['staff_id'] for p in current_route]
                        cluster_group = cluster_group[~cluster_group['staff_id'].isin(assigned_ids)]
                    else:
                        break
                
                if len(cluster_group) > 0:
                    self._assign_remaining_staff(cluster_group, routes)
            
            return routes
            
        except Exception as e:
            st.error(f"Error in route optimization: {str(e)}")
            return {}

    def calculate_route_metrics(self, route):
        """Calculate total distance and cost for a route"""
        if not route:
            return 0, 0
        
        total_distance = 0
        points = [(p['latitude'], p['longitude']) for p in route]
        points.append((self.office_location['lat'], self.office_location['lon']))
        
        for i in range(len(points) - 1):
            distance = geodesic(points[i], points[i + 1]).km
            total_distance += distance
        
        return total_distance, total_distance * self.COST_PER_KM

    def get_route_directions(self, origin, destination, waypoints=None):
        """Get route directions using Google Maps Directions API"""
        try:
            if waypoints:
                waypoints = [f"{point['lat']},{point['lng']}" for point in waypoints]
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    waypoints=waypoints,
                    optimize_waypoints=True,
                    mode="driving",
                    departure_time=datetime.now()
                )
            else:
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    mode="driving",
                    departure_time=datetime.now()
                )

            if directions:
                route = directions[0]
                route_polyline = route['overview_polyline']['points']
                duration = sum(leg['duration']['value'] for leg in route['legs'])
                distance = sum(leg['distance']['value'] for leg in route['legs'])
                
                return {
                    'polyline': route_polyline,
                    'duration': duration,
                    'distance': distance,
                    'directions': directions
                }
            return None
        except Exception as e:
            st.error(f"Error getting directions: {str(e)}")
            return None
    def create_map(self, routes):
        """Create an interactive map with multiple layer controls and satellite imagery"""
        try:
            # Initialize base map
            m = folium.Map(
                location=[self.office_location['lat'], self.office_location['lon']],
                zoom_start=13,
                control_scale=True
            )
            
            # Add multiple tile layers
            folium.TileLayer(
                'cartodbpositron',
                name='Street Map',
                control=True
            ).add_to(m)
            
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite View',
                control=True
            ).add_to(m)
            
            # Create office marker group
            office_group = folium.FeatureGroup(name='Office Location', show=True)
            folium.Marker(
                [self.office_location['lat'], self.office_location['lon']],
                popup=folium.Popup(
                    'Main Office',
                    max_width=200
                ),
                icon=folium.Icon(
                    color='red',
                    icon='building',
                    prefix='fa'
                ),
                tooltip="Office Location"
            ).add_to(office_group)
            office_group.add_to(m)
            
            # Define colors for routes
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
                     'violet', 'darkblue', 'darkgreen', 'cadetblue']
            
            # Create route groups
            for route_idx, (route_name, group) in enumerate(routes.items()):
                color = colors[route_idx % len(colors)]
                route_group = folium.FeatureGroup(
                    name=f"{route_name}",
                    show=True,
                    control=True
                )
                
                # Prepare waypoints for Google Maps
                waypoints = [
                    {'lat': staff['latitude'], 'lng': staff['longitude']}
                    for staff in group
                ]
                
                # Get route from Google Maps
                route_data = self.get_route_directions(
                    f"{waypoints[0]['lat']},{waypoints[0]['lng']}",
                    f"{self.office_location['lat']},{self.office_location['lon']}",
                    waypoints[1:] if len(waypoints) > 1 else None
                )
                
                if route_data:
                    # Add route polyline
                    route_coords = polyline.decode(route_data['polyline'])
                    
                    # Create route path with popup
                    route_line = folium.PolyLine(
                        route_coords,
                        weight=4,
                        color=color,
                        opacity=0.8,
                        popup=folium.Popup(
                            f"""
                            <div style='font-family: Arial; font-size: 12px;'>
                                <b>{route_name}</b><br>
                                Distance: {route_data['distance']/1000:.2f} km<br>
                                Duration: {route_data['duration']/60:.0f} min<br>
                                Passengers: {len(group)}
                            </div>
                            """,
                            max_width=200
                        )
                    )
                    route_line.add_to(route_group)
                    
                    # Add staff markers
                    for idx, staff in enumerate(group, 1):
                        # Create detailed popup content
                        popup_content = f"""
                        <div style='font-family: Arial; font-size: 12px;'>
                            <b>{staff['name']}</b><br>
                            Address: {staff['address']}<br>
                            Stop #{idx}<br>
                            Distance to office: {staff['distance_to_office']:.2f} km<br>
                            Pick-up order: {idx} of {len(group)}
                        </div>
                        """
                        
                        # Add staff marker
                        folium.CircleMarker(
                            location=[staff['latitude'], staff['longitude']],
                            radius=8,
                            popup=folium.Popup(
                                popup_content,
                                max_width=200
                            ),
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            weight=2,
                            tooltip=f"Stop #{idx}: {staff['name']}"
                        ).add_to(route_group)
                
                route_group.add_to(m)
            
            # Add layer controls
            folium.LayerControl(
                position='topright',
                collapsed=False,
                autoZIndex=True
            ).add_to(m)
            
            # Add fullscreen control
            folium.plugins.Fullscreen(
                position='topleft',
                title='Fullscreen',
                title_cancel='Exit fullscreen',
                force_separate_button=True
            ).add_to(m)
            
            # Add search control
            folium.plugins.Search(
                layer=office_group,
                search_label='name',
                position='topleft'
            ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return None

    def get_route_directions(self, origin, destination, waypoints=None):
        """Get route directions using Google Maps Directions API"""
        try:
            if waypoints:
                waypoints = [f"{point['lat']},{point['lng']}" for point in waypoints]
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    waypoints=waypoints,
                    optimize_waypoints=True,
                    mode="driving",
                    departure_time=datetime.now()
                )
            else:
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    mode="driving",
                    departure_time=datetime.now()
                )

            if directions:
                route = directions[0]
                route_polyline = route['overview_polyline']['points']
                duration = sum(leg['duration']['value'] for leg in route['legs'])
                distance = sum(leg['distance']['value'] for leg in route['legs'])
                
                return {
                    'polyline': route_polyline,
                    'duration': duration,
                    'distance': distance,
                    'directions': directions
                }
            return None
        except Exception as e:
            st.error(f"Error getting directions: {str(e)}")
            return None
    def calculate_route_metrics(self, route):
        """
        Calculate total distance and cost for a route
        
        Args:
            route (list): List of dictionaries containing staff information with latitude and longitude
            
        Returns:
            tuple: (total_distance in km, total_cost in currency units)
        """
        if not route:
            return 0, 0
        
        try:
            total_distance = 0
            
            # Convert route points to coordinate pairs
            points = [(p['latitude'], p['longitude']) for p in route]
            
            # Add office location as final destination
            points.append((self.office_location['lat'], self.office_location['lon']))
            
            # Calculate cumulative distance between all consecutive points
            for i in range(len(points) - 1):
                distance = geodesic(points[i], points[i + 1]).km
                total_distance += distance
            
            # Calculate total cost based on distance
            total_cost = total_distance * self.COST_PER_KM
            
            # Get route details from Google Maps for more accurate metrics
            try:
                waypoints = [{'lat': p[0], 'lng': p[1]} for p in points[1:-1]]
                route_data = self.get_route_directions(
                    f"{points[0][0]},{points[0][1]}",
                    f"{points[-1][0]},{points[-1][1]}",
                    waypoints=waypoints if waypoints else None
                )
                
                if route_data:
                    # Use Google Maps distance if available
                    total_distance = route_data['distance'] / 1000  # Convert meters to km
                    total_cost = total_distance * self.COST_PER_KM
            except Exception:
                # Fall back to geodesic distance if Google Maps fails
                pass
            
            return total_distance, total_cost
            
        except Exception as e:
            st.error(f"Error calculating route metrics: {str(e)}")
            return 0, 0

    def get_route_summary(self, route):
        """
        Get a detailed summary of route metrics
        
        Args:
            route (list): List of dictionaries containing staff information
            
        Returns:
            dict: Dictionary containing route metrics
        """
        try:
            distance, cost = self.calculate_route_metrics(route)
            
            return {
                'total_distance': distance,
                'total_cost': cost,
                'passenger_count': len(route),
                'cost_per_passenger': cost / len(route) if route else 0,
                'distance_per_passenger': distance / len(route) if route else 0,
                'start_point': route[0]['address'] if route else None,
                'end_point': 'Office',
                'stops': len(route)
            }
        except Exception as e:
            st.error(f"Error generating route summary: {str(e)}")
            return None

    def calculate_total_metrics(self, routes):
        """
        Calculate total metrics for all routes
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Dictionary containing total metrics
        """
        try:
            total_metrics = {
                'total_distance': 0,
                'total_cost': 0,
                'total_passengers': 0,
                'number_of_routes': len(routes),
                'average_route_distance': 0,
                'average_route_cost': 0,
                'average_passengers_per_route': 0
            }
            
            for route_name, route in routes.items():
                distance, cost = self.calculate_route_metrics(route)
                total_metrics['total_distance'] += distance
                total_metrics['total_cost'] += cost
                total_metrics['total_passengers'] += len(route)
            
            if routes:
                total_metrics['average_route_distance'] = total_metrics['total_distance'] / len(routes)
                total_metrics['average_route_cost'] = total_metrics['total_cost'] / len(routes)
                total_metrics['average_passengers_per_route'] = total_metrics['total_passengers'] / len(routes)
            
            total_metrics['cost_per_passenger'] = (
                total_metrics['total_cost'] / total_metrics['total_passengers']
                if total_metrics['total_passengers'] > 0 else 0
            )
            
            return total_metrics
            
        except Exception as e:
            st.error(f"Error calculating total metrics: {str(e)}")
            return None

    def format_metrics(self, metrics):
        """
        Format metrics for display
        
        Args:
            metrics (dict): Dictionary containing metrics
            
        Returns:
            dict: Dictionary containing formatted metrics
        """
        try:
            formatted = {}
            for key, value in metrics.items():
                if 'distance' in key.lower():
                    formatted[key] = f"{value:.2f} km"
                elif 'cost' in key.lower():
                    formatted[key] = f"GHC{value:.2f}"
                elif 'average' in key.lower():
                    if 'distance' in key.lower():
                        formatted[key] = f"{value:.2f} km"
                    elif 'cost' in key.lower():
                        formatted[key] = f"GHC{value:.2f}"
                    else:
                        formatted[key] = f"{value:.2f}"
                else:
                    formatted[key] = value
            return formatted
            
        except Exception as e:
            st.error(f"Error formatting metrics: {str(e)}")
            return None

    def create_metrics_summary(self, routes):
        """
        Create a complete metrics summary for display
        
        Args:
            routes (dict): Dictionary of routes
            
        Returns:
            dict: Dictionary containing formatted metrics summary
        """
        try:
            metrics = self.calculate_total_metrics(routes)
            if metrics:
                formatted_metrics = self.format_metrics(metrics)
                
                summary = {
                    'Overview': {
                        'Total Routes': metrics['number_of_routes'],
                        'Total Passengers': metrics['total_passengers'],
                        'Total Distance': formatted_metrics['total_distance'],
                        'Total Cost': formatted_metrics['total_cost']
                    },
                    'Averages': {
                        'Average Route Distance': formatted_metrics['average_route_distance'],
                        'Average Route Cost': formatted_metrics['average_route_cost'],
                        'Average Passengers per Route': f"{metrics['average_passengers_per_route']:.1f}",
                        'Cost per Passenger': formatted_metrics['cost_per_passenger']
                    },
                    'Routes': {}
                }
                
                for route_name, route in routes.items():
                    route_summary = self.get_route_summary(route)
                    if route_summary:
                        formatted_route_summary = self.format_metrics(route_summary)
                        summary['Routes'][route_name] = formatted_route_summary
                
                return summary
            return None
            
        except Exception as e:
            st.error(f"Error creating metrics summary: {str(e)}")
            return None   
def _assign_remaining_staff(self, remaining_staff, routes):
        for _, row in remaining_staff.iterrows():
            best_route = None
            min_detour = float('inf')
            
            for route_name, route_group in routes.items():
                if len(route_group) < self.MAX_PASSENGERS:
                    current_distance = self.calculate_route_metrics(route_group)[0]
                    test_route = route_group + [row.to_dict()]
                    new_distance = self.calculate_route_metrics(test_route)[0]
                    detour = new_distance - current_distance
                    
                    if detour < min_detour:
                        min_detour = detour
                        best_route = route_name
            
            if best_route:
                routes[best_route].append(row.to_dict())

def calculate_route_metrics(self, route):
        if not route:
            return 0, 0
            
        total_distance = 0
        points = [(p['latitude'], p['longitude']) for p in route]
        points.append((self.office_location['lat'], self.office_location['lon']))
        
        for i in range(len(points) - 1):
            distance = geodesic(points[i], points[i + 1]).km
            total_distance += distance
            
        return total_distance, total_distance * self.COST_PER_KM

def get_route_directions(self, origin, destination, waypoints=None):
        try:
            if waypoints:
                waypoints = [f"{point['lat']},{point['lng']}" for point in waypoints]
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    waypoints=waypoints,
                    optimize_waypoints=True,
                    mode="driving",
                    departure_time=datetime.now()
                )
            else:
                directions = self.gmaps.directions(
                    origin,
                    destination,
                    mode="driving",
                    departure_time=datetime.now()
                )

            if directions:
                route = directions[0]
                route_polyline = route['overview_polyline']['points']
                duration = sum(leg['duration']['value'] for leg in route['legs'])
                distance = sum(leg['distance']['value'] for leg in route['legs'])
                
                return {
                    'polyline': route_polyline,
                    'duration': duration,
                    'distance': distance,
                    'directions': directions
                }
            return None
        except Exception as e:
            st.error(f"Error getting directions: {str(e)}")
            return None

def create_map(self, routes):
        try:
            m = folium.Map(
                location=[self.office_location['lat'], self.office_location['lon']],
                zoom_start=13
            )
            
            # Add tile layers
            folium.TileLayer('cartodbpositron', name='Street Map').add_to(m)
            folium.TileLayer(
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite'
            ).add_to(m)
            
            # Create office marker group
            office_group = folium.FeatureGroup(name='Office Location', show=True)
            folium.Marker(
                [self.office_location['lat'], self.office_location['lon']],
                popup='Office',
                icon=folium.Icon(color='red', icon='building', prefix='fa'),
                tooltip="Office Location"
            ).add_to(office_group)
            office_group.add_to(m)
            
            colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'violet', 'pink','aqua','yellow']
            
            for route_idx, (route_name, group) in enumerate(routes.items()):
                color = colors[route_idx % len(colors)]
                route_group = folium.FeatureGroup(name=route_name, show=True)
                
                waypoints = [{'lat': staff['latitude'], 'lng': staff['longitude']} for staff in group]
                route_data = self.get_route_directions(
                    f"{waypoints[0]['lat']},{waypoints[0]['lng']}",
                    f"{self.office_location['lat']},{self.office_location['lon']}",
                    waypoints[1:] if len(waypoints) > 1 else None
                )
                
                if route_data:
                    route_coords = polyline.decode(route_data['polyline'])
                    folium.PolyLine(
                        route_coords,
                        weight=3,
                        color=color,
                        opacity=0.8,
                        popup=f"{route_name}<br>Distance: {route_data['distance']/1000:.2f} km<br>"
                              f"Duration: {route_data['duration']/60:.0f} min"
                    ).add_to(route_group)
                    
                    for idx, staff in enumerate(group, 1):
                        folium.CircleMarker(
                            [staff['latitude'], staff['longitude']],
                            radius=8,
                            popup=folium.Popup(
                                f"""
                                <b>{staff['name']}</b><br>
                                Address: {staff['address']}<br>
                                Stop #{idx}<br>
                                Distance to office: {staff['distance_to_office']:.2f} km
                                """,
                                max_width=200
                            ),
                            color=color,
                            fill=True,
                            fill_opacity=0.7,
                            tooltip=f"Stop #{idx}: {staff['name']}"
                        ).add_to(route_group)
                
                route_group.add_to(m)
            
            folium.LayerControl(
                position='topright',
                collapsed=False
            ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return None

# Part 3: UI Helper Functions
def load_css():
    st.markdown(
        """
        <meta charset="UTF-8">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <link rel="icon" href="https://www.4th-ir.com/favicon.ico">
        
        <title>4thir-POC-repo</title>
        <meta name="title" content="4thir-POC-repo" />
        <meta name="description" content="view our proof of concepts" />

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="og:title" content="4thir-POC-repo" />
        <meta property="og:description" content="view our proof of concepts" />
        <meta property="og:image" content="https://www.4th-ir.com/favicon.ico" />

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image" />
        <meta property="twitter:url" content="https://4thir-poc-repositoty.streamlit.app/" />
        <meta property="twitter:title" content="4thir-POC-repo" />
        <meta property="twitter:description" content="view our proof of concepts" />
        <meta property="twitter:image" content="https://www.4th-ir.com/favicon.ico" />

        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        """,
        unsafe_allow_html=True,
    )

    
    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
                background-color: #4267B2;
                color: white;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: scale(1.02);
            }
            .metric-card {
               
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
                
            }
            .search-box {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_navbar():
    st.markdown(
        """
       <nav class="navbar fixed-top navbar-expand-lg navbar-dark bg-white text-bold shadow-sm">
            <a class="navbar-brand text-primary" href="#" target="_blank">
                <img src="https://cdn.bio.link/uploads/profile_pictures/2022-01-27/Q0mOblteBj6VooKH4zNCa9zKD5JHkVnM.png" alt="4th-ir logo" style='width:50px'>
               Ride Router
            </a>
        </nav>

   

        """,
        unsafe_allow_html=True
    )

def show_metrics_dashboard(metrics):
    st.markdown("""
        <div class="card" style="padding: 1rem;  border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin-bottom: 1rem;">Route Metrics Dashboard</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
    """, unsafe_allow_html=True)
    
    for key, value in metrics.items():
        st.markdown(f"""
            <div class="metric-card card">
                <h4>{key.replace('_', ' ').title()}</h4>
                <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Part 4: Main Application
def main():
    
    
    load_css()
    create_navbar()
    
    # Initialize session state
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    if 'staff_data' not in st.session_state:
        st.session_state.staff_data = None
    if 'routes' not in st.session_state:
        st.session_state.routes = None

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("1. Data Input")
        data_option = st.radio(
            "Choose data input method",
            ["Upload CSV", "Use Sample Data"]
        )
        
        optimizer = StaffTransportOptimizer(google_maps_key=os.getenv("GOOGLE_MAPS_API_KEY"))
        
        if data_option == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload staff locations CSV",
                type=["csv"],
                help="""
                Required columns:
                - staff_id (unique identifier)
                - name (staff name)
                - latitude (valid coordinate)
                - longitude (valid coordinate)
                - address (location description)
                """
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.staff_data = optimizer.validate_staff_data(df)
                    st.success("✅ Data validated successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.staff_data = None
        else:
            if st.button("Load Sample Data"):
                st.session_state.staff_data = optimizer.load_sample_data()
                st.success("✅ Sample data loaded successfully!")
        
        st.subheader("2. Route Parameters")
        grid_size = st.slider(
            "Cluster Grid Size",
            min_value=2,
            max_value=5,
            value=3,
            help="Controls the number of potential clusters"
        )
        
        sigma = st.slider(
            "Cluster Radius",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls how spread out the clusters are"
        )
        
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Controls how quickly the clustering algorithm adapts"
        )
        
        if st.button(" Optimize Routes", type="primary"):
            if st.session_state.staff_data is not None:
                with st.spinner("Optimizing routes..."):
                    try:
                        clustered_data = optimizer.create_clusters(
                            st.session_state.staff_data,
                            grid_size=grid_size,
                            sigma=sigma,
                            learning_rate=learning_rate
                        )
                        
                        if clustered_data is not None:
                            st.session_state.routes = optimizer.optimize_routes(clustered_data)
                            if st.session_state.routes:
                                st.session_state.optimization_done = True
                                st.success("✅ Routes optimized successfully!")
                            else:
                                st.error("❌ Route optimization failed. Try different parameters.")
                        else:
                            st.error("❌ Clustering failed. Try different parameters.")
                    except Exception as e:
                        st.error(f"❌ Optimization error: {str(e)}")
            else:
                st.warning("⚠️ Please upload valid staff data first.")

    # Main content area
    if st.session_state.staff_data is not None:
        col1, col2, = st.columns([2, 1])
        
        with col1:
            if st.session_state.optimization_done:
                st.subheader(" Route Map")
                m = optimizer.create_map(st.session_state.routes)
                if m is not None:
                    st_folium(m, width=800, height=600)
                    
                    st.info("""
                    **Map Controls:**
                    - Use the layer control (top right) to toggle routes and map type
                    - Click on markers for detailed information
                    - Zoom in/out using the mouse wheel or +/- buttons
                    """)
        
        with col2:
            st.subheader("Staff Directory")
            search_term = st.text_input("Search staff by name or address")
            
            display_df = st.session_state.staff_data[['name', 'address','latitude','longitude']].copy()
            if search_term:
                mask = (
                    display_df['name'].str.contains(search_term, case=False) |
                    display_df['address'].str.contains(search_term, case=False)
                    
                )
                display_df = display_df[mask]
            
            st.dataframe(display_df, height=300)
            
            if st.session_state.optimization_done:
                st.subheader("Route Details")
                
                metrics = {
                    'total_distance': 0,
                    'total_cost': 0,
                    'total_duration': 0
                }
                
                for route_name, route in st.session_state.routes.items():
                    with st.expander(f" {route_name}"):
                        distance, cost = optimizer.calculate_route_metrics(route)
                        metrics['total_distance'] += distance
                        metrics['total_cost'] += cost
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Distance", f"{distance:.2f} km")
                        with col2:
                            st.metric("Cost", f"GHC{cost:.2f}")
                        
                        st.dataframe(
                            pd.DataFrame(route)[['name', 'address', 'distance_to_office']],
                            height=200
                        )
                
                show_metrics_dashboard({
                    'Total Distance': f"{metrics['total_distance']:.2f} km",
                    'Total Cost': f"GHC{metrics['total_cost']:.2f}",
                    'Number of Routes': len(st.session_state.routes),
                    'Average Cost/Route': f"GHC{metrics['total_cost']/len(st.session_state.routes):.2f}"
                })

if __name__ == "__main__":
    main()