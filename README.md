# Staff Transport Optimization System

## Overview
This Streamlit application optimizes staff transportation by clustering employees based on location and generating efficient routes using Google Maps API. It incorporates Self-Organizing Maps (SOM) for clustering and Folium for interactive map visualizations.

## Features
- **Upload or Use Sample Data**: Upload a CSV file with staff location data or use generated sample data.
- **Data Validation**: Cleans and validates staff location data before processing.
- **Clustering Using SOM**: Groups staff members based on their geographic location.
- **Route Optimization**: Uses Google Maps API to find the most efficient routes.
- **Interactive Map Visualization**: Displays optimized routes with waypoints and distances.
- **Metrics Dashboard**: Summarizes total distance, cost, and other route metrics.

## Technologies Used
- **Streamlit**: Web-based user interface.
- **Folium**: Interactive map rendering.
- **Google Maps API**: Route and distance calculations.
- **Scikit-learn**: Data normalization.
- **Geopy**: Distance calculations between locations.
- **Numpy & Pandas**: Data processing and manipulation.
- **SOM (Self-Organizing Maps)**: Clustering algorithm for grouping staff locations.

## Installation
Ensure you have Python 3.7+ installed and install dependencies using:

```sh
pip install streamlit pandas folium streamlit_folium numpy geopy googlemaps scikit-learn python-dotenv polyline
```

## Environment Variables
Create a `.env` file and add your Google Maps API key:

```
GOOGLE_MAPS_API_KEY=your_api_key_here
```

## Running the Application
Run the Streamlit application using:

```sh
streamlit run app.py
```

## Usage Guide
1. **Load Staff Data**:
   - Upload a CSV file with columns: `staff_id`, `name`, `latitude`, `longitude`, `address`.
   - Alternatively, load sample data.
2. **Set Clustering Parameters**:
   - Adjust grid size, sigma, and learning rate for clustering.
3. **Optimize Routes**:
   - Click "Optimize Routes" to generate optimal transport routes.
4. **View Map & Metrics**:
   - Explore the interactive map and summary statistics.

## Folder Structure
```
.
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── .env                 # API key storage (ignored in Git)
```

## Notes
- Ensure Google Maps API key is valid for routing services.
- Data must have valid latitude and longitude values within Ghana's coordinates.

## License
This project is licensed under the MIT License.

