from src.logger import logging
from src.exception import CustomException
import sys

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut

import numpy as np
import pandas as pd
import time

class DataPreprocessor:
    def __init__(self):
        pass

    def initiate_preprocessing(self,X:pd.DataFrame):
        city_distance_transformer = CityDistanceTransformer()
        date_time = Transform_datetime()
        other_transformations = CustomTransformer()
        X = city_distance_transformer.transform(X)
        X = date_time.transform(X)
        X = other_transformations.transform(X)

        return X

class CityDistanceTransformer:
    def __init__(self,col1='Source',col2='Destination',newcol='Distance_km',user_agent='Small_data_science_project'):
        """
        Parameters:
        - city_col1: Name of first city column (default: 'Source')
        - city_col2: Name of second city column (default: 'Destination')
        - new_col: Name for new distance column (default: 'Distance_km')
        - user_agent: Unique identifier for geocoding API 
        """
        self.col1 = col1
        self.col2 = col2
        self.newcol = newcol
        self.user_agent = user_agent
        self.city_coords = {}

    def geocode(self, city_name: str, max_retries=3, timeout=10):
        geolocator = Nominatim(user_agent=self.user_agent)
        for _ in range(max_retries):
            try:
                location = geolocator.geocode(city_name, timeout=timeout)
                if location:
                    return location
            except (GeocoderTimedOut, Exception) as e:
                logging.warning(f"Retrying {city_name}... (Error: {str(e)})")
                time.sleep(2)
        logging.error(f"Failed to geocode {city_name} after {max_retries} retries")
        return None
        
    
    def distance(self,row):
        source = row[self.col1]
        destination = row[self.col2]
        coords1 = self.city_coords.get(source)
        coords2 = self.city_coords.get(destination)

        if coords1 and coords2:
            return geodesic((coords1.latitude,coords1.longitude),(coords2.latitude,coords2.longitude)).km
        return np.nan
    
    
    def transform(self,X:pd.DataFrame,y=None):
        X = X.copy()

        cities = set(X[self.col1]).union(set(X[self.col2]))
        
        for city in cities:
            if city not in self.city_coords:
                self.city_coords[city] = self.geocode(city_name=city)
        logging.info('All cities coordinates found and stored')

        X[self.newcol] = X.apply(self.distance,axis=1).astype(float)
        logging.info('All distances calculated and updated')
        return X

class Transform_datetime:
    def __init__(self,date_col='Date_of_Journey',dep_col='Dep_Time',arr_col='Arrival_Time',duration_col='Duration'):
        """
        Parameters:
        - date_col: Name of column containing Date of journey (default: 'Date_of_Journey')
        - dep_col: Name of column containing departure time (default: 'Dep_Time')
        - new_col: Name of column containing arrival time (default: 'Arrival_Time')
        - duration_col: Name of column containing duration of flight(default: 'Duration') 
        """
        
        self.date_col=date_col
        self.dep_col=dep_col
        self.arr_col=arr_col
        self.duration_col=duration_col
    
    def transform(self,df:pd.DataFrame,y=None):
        df = df.copy()

        # code for transforming date of journey
        df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"],format='%d/%m/%Y',dayfirst=True)
        df['Day_of_journey']  = df['Date_of_Journey'].dt.day
        df['Month_of_journey']  = df['Date_of_Journey'].dt.month
        df['Year_of_journey']  = df['Date_of_Journey'].dt.year
        
        logging.info('Date of Journey transformation completed')

        # code for transforming departure time
        df['Dep_hour'] = df[self.dep_col].str.split(':').str[0]
        df['Dep_minute'] = df[self.dep_col].str.split(':').str[1]
        df['Dep_hour'] = df['Dep_hour'].astype(int)
        df['Dep_minute'] = df['Dep_minute'].astype(int)
        df['Dep_time_sinceMidnight'] = df['Dep_hour']*60 + df['Dep_minute']
        logging.info('Departure time transformation completed')

        # code for transforming arrival time
        df[self.arr_col] = df[self.arr_col].str.split(" ").str[0]
        df['Arrival_hour'] = df[self.arr_col].str.split(":").str[0]
        df['Arrival_minute'] = df[self.arr_col].str.split(":").str[1]
        df['Arrival_hour'] = df['Arrival_hour'].astype(int)
        df['Arrival_minute'] = df['Arrival_minute'].astype(int)
        df['Arrival_time_sinceMidnight'] = df['Arrival_hour']*60+df['Arrival_minute']
        logging.info('Arrival time transformation completed')

        #code for transforming duration
        duration = pd.to_timedelta(df['Duration'])
        for i  in range(0,len(duration)):
            df.loc[i,'Duration_mins'] = duration[i].seconds //60
        logging.info('Duration transformation completed')
        return df.drop(columns=['Date_of_Journey', 'Source', 'Destination', 'Dep_Time',
                           'Arrival_Time', 'Duration', 'Year_of_journey','Route'], errors='ignore')
    
class CustomTransformer:
    def __init__(self,air_col='Airline',tot='Total_Stops',Add_info='Additional_Info'):
        self.air_col=air_col
        self.tot=tot
        self.Add_info=Add_info

    def airline(self,x):
        if 'Trujet' in x or 'GoAir' in x or 'Multiple carriers' in x:
            return 'Multiple carriers'
        elif 'Jet Airways' in x:
            return 'Jet Airways'
        elif 'Vistara' in x:
            return 'Vistara'
        else: 
            return x

    def transform(self, X:pd.DataFrame, y = None):
        X['Premium'] = X[self.air_col].apply(lambda x: 1 if 'Premium' in x or 'Business' in x else 0)
        X[self.air_col] = X[self.air_col].apply(self.airline)
        X['Total_Stops'] = X['Total_Stops'].map({'non-stop':0,'2 stops':2,'1 stop':1,'3 stops':3,'4 stops':4})
        X['Total_Stops'] = X['Total_Stops'].astype(int)
        X['Additional_Info'] = X['Additional_Info'].apply(lambda x: 'No Info' if 'No info' in x else x)
        logging.info('Transformation of Airline names,Additional Info and Total stops completed')

        return X