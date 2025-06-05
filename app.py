import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

class UberPuneAnalysis:
    def __init__(self, data_path='pune_uber_rides.csv'):
        self.df = None  # Initialize as None
        try:
            self.df = pd.read_csv(data_path)
            self.preprocess_data()
        except FileNotFoundError:
            st.error("Data file not found. Please provide a valid file path.")
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")

    def preprocess_data(self):
        if self.df is None:
            st.error("Dataframe is not initialized.")
            return

        # Convert datetime column
        self.df['pickup_datetime'] = pd.to_datetime(self.df['pickup_datetime'], errors='coerce')

        # Filter rows with valid datetime values
        self.df = self.df.dropna(subset=['pickup_datetime'])

        # Create derived columns
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self.df['day'] = self.df['pickup_datetime'].dt.day_name()
        self.df['month'] = self.df['pickup_datetime'].dt.month
        self.df['weekday'] = self.df['pickup_datetime'].dt.weekday

        # Handle missing coordinate values
        self.df = self.df.dropna(subset=['pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon'])

        # Calculate trip distances
        self.df['distance'] = self.calculate_distance()

        # Clean up fare amounts
        self.df['fare_amount'] = pd.to_numeric(self.df['fare_amount'], errors='coerce')
        self.df = self.df.dropna(subset=['fare_amount'])
     # calculating distance
    def calculate_distance(self):
        from math import radians, sin, cos, sqrt, atan2

        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Radius of Earth in kilometers
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        return self.df.apply(lambda row: haversine_distance(
            row['pickup_lat'], row['pickup_lon'],
            row['dropoff_lat'], row['dropoff_lon']
        ), axis=1)

    def generate_summary_stats(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        return {
            'total_rides': len(self.df),
            'average_fare': self.df['fare_amount'].mean(),
            'average_distance': self.df['distance'].mean(),
            'busiest_hour': self.df['hour'].mode().iloc[0] if not self.df['hour'].empty else None,
            'busiest_day': self.df['day'].mode().iloc[0] if not self.df['day'].empty else None,
        }

    def analyze_peak_hours(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        hourly_rides = self.df['hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=hourly_rides.index, y=hourly_rides.values, ax=ax)
        ax.set_title('Hourly Distribution of Uber Rides in Pune')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Rides')
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.3)
        return fig

    def analyze_weekly_pattern(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_rides = self.df['day'].value_counts().reindex(days_order)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=weekly_rides.index, y=weekly_rides.values, ax=ax)
        ax.set_title('Weekly Distribution of Uber Rides in Pune')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Rides')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
        return fig

    def analyze_popular_locations(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12, tiles='CartoDB positron')
        heat_data = [[row['pickup_lat'], row['pickup_lon']] for _, row in self.df.iterrows()]
        plugins.HeatMap(heat_data, radius=15).add_to(pune_map)

        geolocator = Nominatim(user_agent="uber_pune_analysis")
        popular_locations = self.df.groupby(['pickup_lat', 'pickup_lon']).size().reset_index(name='count')
        popular_locations = popular_locations[popular_locations['count'] > 10]

        for _, row in popular_locations.iterrows():
            try:
                location = geolocator.reverse((row['pickup_lat'], row['pickup_lon']), timeout=10)
                address = location.address if location else "Unknown Location"
            except GeocoderTimedOut:
                address = "Geocoding Timed Out"

            folium.CircleMarker(
                location=[row['pickup_lat'], row['pickup_lon']],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"Count: {row['count']}\n{address}"
            ).add_to(pune_map)

        map_file = 'pune_map.html'
        pune_map.save(map_file)
        return map_file

    def analyze_fare_distribution(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(self.df['fare_amount'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Fare Amounts')
        ax.set_xlabel('Fare Amount (₹)')
        ax.set_ylabel('Frequency')
        return fig

    def analyze_trip_durations(self):
        if self.df is None:
            st.error("Dataframe is not loaded.")
            return None

        trip_durations = self.df['distance'] / 20 * 60
        self.df['trip_duration'] = trip_durations
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(self.df['trip_duration'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Trip Durations')
        ax.set_xlabel('Trip Duration (minutes)')
        ax.set_ylabel('Frequency')
        return fig


def main():
    st.set_page_config(page_title='Uber Pune Analysis Dashboard', layout='wide')
    st.title('Uber Rides Analysis - Pune')

    analyzer = UberPuneAnalysis()

    if analyzer.df is not None:
        summary_stats = analyzer.generate_summary_stats()

        with st.container():
            col1, col2, col3 = st.columns(3)
            col1.metric('Total Rides', f"{summary_stats['total_rides']:,}")
            col2.metric('Average Fare', f"₹{summary_stats['average_fare']:.2f}")
            col3.metric('Average Distance', f"{summary_stats['average_distance']:.2f} km")

        st.subheader('Peak Hours Analysis')
        st.pyplot(analyzer.analyze_peak_hours())

        st.subheader('Weekly Pattern Analysis')
        st.pyplot(analyzer.analyze_weekly_pattern())

        st.subheader('Popular Locations')
        map_file = analyzer.analyze_popular_locations()
        with open(map_file, 'r') as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=600)

        st.subheader('Trip Duration Analysis')
        st.pyplot(analyzer.analyze_trip_durations())

        st.subheader('Fare Distribution')
        st.pyplot(analyzer.analyze_fare_distribution())


if __name__ == '__main__':
    main()
