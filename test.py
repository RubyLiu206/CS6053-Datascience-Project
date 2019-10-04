# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:41:03 2019

@author: ruby_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
def data_cleaning(dataframe):
    dataframe = dataframe.dropna()





"""
data preprocessing:
    deal with the raw data 
    deal with the NaN data
"""
    
def cleaning_dataframe(row):
    try:
        return float(row)
    except:
        return np.NaN
    
    
    
"""
data preprocessing:
    create a new dataset with the datetime
    choosing the rows which need for the table
    

"""
def preprocess_temperature(NASA_temperature):
    # using a datatime index
    #basic manipulation and dealing with missing values
    #resampling to a diffeent frequency
    range_date = pd.date_range(start = '1/1/1880', end = '1/03/2019', freq = 'M')
    #print(type(range_date))
    table = pd.DataFrame(range_date, columns = ['date'])
    table['average_temperature_monthly'] = None
    table.set_index('date', inplace = True)
    # only use the first 13th columns, and leave the season line
    NASA_temperature = NASA_temperature.iloc[:,:13]
    return NASA_temperature, table    


def preprocess_CO2(CO2_emission):
    range_date = pd.date_range(start = '31/12/1960', end = '31/12/2018', freq = 'Y')
    #print(type(range_date))
    table_CO2 = pd.DataFrame(range_date, columns = ['date'])
    #print(table_CO2.index)
    
    CO2_emission = CO2_emission[CO2_emission['Country Name']=='World'].loc[:,'1960':'2018']
    CO2_emission = CO2_emission.T
    CO2_emission.columns = ['value']
    
    #print(table_CO2)
    return CO2_emission, table_CO2


"""
populate dataset:
    translating the dataset from raw data to the datetime dataset
    lambda function 

"""

def populate_CO2(row, CO2_emission):
    index = str(row['date'].year)
    value= CO2_emission.loc[index]
    return value


def populate_df_with_anomolies_from_row(row,table):
    year = row['Year']
    # Anomaly values (they seem to be a mixture of strings and floats)
    monthly_anomolies = row.iloc[1:]
    # Abbreviated month names (index names)
    months = monthly_anomolies.index
    for month in monthly_anomolies.index:
        # monthrange return the day in the specified year and month.
        # eg. max_day = calendar.monthrange(2001, month)[1]
        last_day = calendar.monthrange(year,datetime.strptime(month, '%b').month)[1]
        # construct the index with which we can reference our new DataFrame (to populate) 
        # dateformat for date
        date_index = datetime.strptime(f'{year} {month} {last_day}', '%Y %b %d')
        # put the value in row to the table loc index
        table.loc[date_index] = monthly_anomolies[month]
      
   


def correlation(GlobalTemperatures):
    corr = GlobalTemperatures.LandAverageTemperature.corr(GlobalTemperatures['LandAndOceanAverageTemperature'])
    return corr


"""
plot function
    normal plot
    resampling plot
"""


def plot_temperature(table):
    plt.figure(figsize = (20,8))
    plt.xlabel('Time')
    plt.ylabel('Temperature at the specified time')
    plt.plot(table, color = 'blue', linewidth = 1.0)

# Resampling or converting a time series to a particular frequency
def resample_plot_temperature(table):
    table = table.resample('A').mean()
    #print(table)
    plt.figure(figsize = (20,8))
    plt.xlabel('Time')
    plt.ylabel('Temperature at the specified time')
    plt.plot(table, color = 'blue', linewidth = 1.0)

def plot_CO2(table):
    plt.figure(figsize = (20,8))
    plt.xlabel('Time')
    plt.ylabel('co2 emission at the specified time')
    plt.plot(table, color = 'blue', linewidth = 1.0)

# Resampling or converting a time series to a particular frequency
def resample_plot_CO2(table):
    table = table.resample('A').mean()
    print(table)
    plt.figure(figsize = (20,8))
    plt.xlabel('Time')
    plt.ylabel('co2 emission at the specified time')
    plt.plot(table, color = 'blue', linewidth = 1.0)
    
    
    
"""
main 
"""
def main():
    # data from GlobalTemperatures have: date, LandAverageTemperature, LandAverageTemperatureUncertainty
    # LandMaxTemperature, LandMaxTemperatureUncertainty, LandMinTemperature, LandMinTemperatureUncertainty
    # LandAndOceanAverageTemperature, LandAndOceanAverageTemperatureUncertai
    """
    GlobalTemperatures = pd.read_csv("GlobalTemperatures.csv")
    print(type(GlobalTemperatures))
    GlobalTemperatures_columns = GlobalTemperatures.columns
    print(GlobalTemperatures_columns)
    GlobalTemperatures = GlobalTemperatures.dropna()
    temperatures = GlobalTemperatures.drop('dt',1)
    corr_GlobalTemperatures = temperatures.corr(method ='pearson')
    corr = temperatures.LandAverageTemperature.corr(temperatures['LandAndOceanAverageTemperature'])
    # corr = 0.9880655 which means the correlation between LandAverageTemperature and LandAndOceanAverageTemperature is positive, and high
    print(corr_GlobalTemperatures)
    #correlation(GlobalTemperatures)
    """
    
    
    #global temperature from NASA
    NASA_temperature = pd.read_csv('data/GLB.Ts+dSST.csv', skiprows=1)
    NASA_temperature.head()
    
    
    # data preprocessing
    NASA_temperature, table = preprocess_temperature(NASA_temperature)
    _ = NASA_temperature.apply(lambda row: populate_df_with_anomolies_from_row(row,table), axis=1)
    #print(table)
    
    table['average_temperature_monthly'] = table['average_temperature_monthly'].apply(lambda row: cleaning_dataframe(row))
    table.fillna(method='ffill', inplace=True)
    plot_temperature(table)
    resample_plot_temperature(table)
    
    
    # CO2 emission from world bank
    
    CO2_emission = pd.read_csv('data/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_248248.csv', skiprows=3)
    CO2_emission.head()
    CO2_emission, table_CO2 = preprocess_CO2(CO2_emission)
    v = table_CO2.apply(lambda row: populate_CO2(row, CO2_emission), axis=1)
    table_CO2['Global CO2 Emissions per Capita'] = v
    table_CO2.set_index('date',inplace = True)
    table_CO2.fillna(method='ffill', inplace=True)

    plot_CO2(table_CO2)
    resample_plot_CO2(table_CO2)
"""
    GlobalTemperatures_Country = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
    print(type(GlobalTemperatures_Country))
    GlobalTemperatures_Country_columns = GlobalTemperatures_Country.columns
    #print(GlobalTemperatures_Country_columns)
    
    GlobalTemperatures_City = pd.read_csv("GlobalLandTemperaturesByCity.csv")
    print(type(GlobalTemperatures_City))
    GlobalTemperatures_City_columns = GlobalTemperatures_City.columns
    #print(GlobalTemperatures_City_columns)
    
    GlobalTemperatures_MajorCity = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv")
    print(type(GlobalTemperatures_MajorCity))
    GlobalTemperatures_MajorCity_columns = GlobalTemperatures_MajorCity.columns
    #print(GlobalTemperatures_MajorCity_columns)
    
    GlobalTemperatures_State = pd.read_csv("GlobalLandTemperaturesByState.csv")
    print(type(GlobalTemperatures_State))
    GlobalTemperatures_State_columns = GlobalTemperatures_State.columns
    #print(GlobalTemperatures_State_columns)
"""
main()


