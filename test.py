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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import time
import warnings
import statsmodels.api as sm


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
plot the global ball temperature changing
plot the bar for all countries temperature from high to low

"""
def print_global_plot(global_temp_country,countries,mean_temp):
   
    data = [ dict(
            type = 'choropleth',
            locations = countries,
            z = mean_temp,
            locationmode = 'country names',
            text = countries,
            marker = dict(
                line = dict(color = 'rgb(0,0,0)', width = 1)),
                colorbar = dict(autotick = True, tickprefix = '', 
                title = '# Average\nTemperature,\n°C')
                )
           ]
    layout = dict(
        title = 'Average land temperature in countries',
        geo = dict(
            showframe = False,
            showocean = True,
            oceancolor = 'rgb(0,255,255)',
            projection = dict(
            type = 'orthographic',
                rotation = dict(
                        lon = 60,
                        lat = 10),
            ),
            lonaxis =  dict(
                    showgrid = True,
                    gridcolor = 'rgb(102, 102, 102)'
                ),
            lataxis = dict(
                    showgrid = True,
                    gridcolor = 'rgb(102, 102, 102)'
                    )
                ),
            )
    
    fig = dict(data=data, layout=layout)
    py.plot(fig, validate=False, filename='worldmap') 
    
    # bar plot for all country with desc order
    mean_temp_bar, countries_bar = (list(x) for x in zip(*sorted(zip(mean_temp, countries), reverse = True)))
    sns.set(font_scale=0.9) 
    f, ax = plt.subplots(figsize=(4.5, 50))
    colors_cw = sns.color_palette('coolwarm', len(countries))
    sns.barplot(mean_temp_bar, countries_bar, palette = colors_cw[::-1])
    Text = ax.set(xlabel='Average temperature', title='Average land temperature in countries')


def Global_flat(global_temp_country,global_temp_country_clear,countries):
    #Extract the year from a date
    years = np.unique(global_temp_country_clear['dt'].apply(lambda x: x[:4]))
    mean_temp_year_country = [ [0] * len(countries) for i in range(len(years[::10]))]
    mean_temp = []
    for country in countries:
        mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == country]['AverageTemperature'].mean())
       
    j = 0
    for country in countries:
        all_temp_country = global_temp_country_clear[global_temp_country_clear['Country'] == country]
        i = 0
        for year in years[::10]:
            mean_temp_year_country[i][j] = all_temp_country[all_temp_country['dt'].apply(
                    lambda x: x[:4]) == year]['AverageTemperature'].mean()
            i +=1
        j += 1

    data = [ dict(
            type = 'choropleth',
            locations = countries,
            z = mean_temp,
            locationmode = 'country names',
            text = countries,
            marker = dict(
                line = dict(color = 'rgb(0,0,0)', width = 1)),
                colorbar = dict(autotick = True, tickprefix = '',
                title = '# Average\nTemperature,\n°C'),
                )
           ]
    
    layout = dict(
        title ='Countries Average land temperature',
        geo = dict(
            showframe = False,
            showocean = True,
            oceancolor = 'rgb(0,255,255)',
            type = 'equirectangular'
        ),
    )
    
    fig = dict(data=data, layout=layout)
    py.plot(fig, validate=False, filename='world_temp_map')





def Global_temp(global_temp,global_temp_country_clear):
  
    #Extract the year from a date
    years = np.unique(global_temp['dt'].apply(lambda x: x[:4]))
    mean_temp_world = []
    mean_temp_world_uncertainty = []
    
    for year in years:
        mean_temp_world.append(global_temp[global_temp['dt'].apply(lambda x: x[:4]) == year]['LandAverageTemperature'].mean())
        mean_temp_world_uncertainty.append(global_temp[global_temp['dt'].apply(lambda x: x[:4]) == year]['LandAverageTemperatureUncertainty'].mean())
    trace0 = go.Scatter(
        x = years, 
        y = np.array(mean_temp_world) + np.array(mean_temp_world_uncertainty),
        fill= None,
        mode='lines',
        name='Uncertainty top',
        line=dict(
            color='rgb(0, 255, 255)',
        )
    )
    trace1 = go.Scatter(
        x = years, 
        y = np.array(mean_temp_world) - np.array(mean_temp_world_uncertainty),
        fill='tonexty',
        mode='lines',
        name='Uncertainty bot',
        line=dict(
            color='rgb(0, 255, 255)',
        )
    )
    
    trace2 = go.Scatter(
        x = years, 
        y = mean_temp_world,
        name='Average Temperature',
        line=dict(
            color='rgb(199, 121, 093)',
        )
    )
    data = [trace0, trace1, trace2]
    
    layout = go.Layout(
        xaxis=dict(title='year'),
        yaxis=dict(title='Average Temperature, °C'),
        title='Average land temperature in world',
        showlegend = False)
    
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
    
    
    
""" specific continent """
def average_tempurature_country(global_temp_country_clear):
    continent = ['Russia', 'United States', 'Niger', 'Greenland', 'Australia', 'Bolivia']
    mean_temp_year_country = [ [0] * len(years[70:]) for i in range(len(continent))]
    j = 0
    for country in continent:
        all_temp_country = global_temp_country_clear[global_temp_country_clear['Country'] == country]
        i = 0
        for year in years[70:]:
            mean_temp_year_country[j][i] = all_temp_country[all_temp_country['dt'].apply(
                    lambda x: x[:4]) == year]['AverageTemperature'].mean()
            i +=1
        j += 1
    
    traces = []
    colors = ['rgb(0, 255, 255)', 'rgb(255, 0, 255)', 'rgb(0, 0, 0)',
              'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)']
    for i in range(len(continent)):
        traces.append(go.Scatter(
            x=years[70:],
            y=mean_temp_year_country[i],
            mode='lines',
            name=continent[i],
            line=dict(color=colors[i]),
        ))
    
    layout = go.Layout(
        xaxis=dict(title='year'),
        yaxis=dict(title='Average Temperature, °C'),
        title='Average land temperature on the continents',)
    
    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig)




"""
high_view :
from continent to country temperature analysis
data preprocessing
data analysis
"""
def high_view():
    # data from GlobalTemperatures have: date, LandAverageTemperature, LandAverageTemperatureUncertainty
    # LandMaxTemperature, LandMaxTemperatureUncertainty, LandMinTemperature, LandMinTemperatureUncertainty
    # LandAndOceanAverageTemperature, LandAndOceanAverageTemperatureUncertai

    
    """global temperature from NASA """
    NASA_temperature = pd.read_csv('data/GLB.Ts+dSST.csv', skiprows=1)
    NASA_temperature.head()
    
    
    """ data preprocessing """
    NASA_temperature, table = preprocess_temperature(NASA_temperature)
    _ = NASA_temperature.apply(lambda row: populate_df_with_anomolies_from_row(row,table), axis=1)
    #print(table)
    
    table['average_temperature_monthly'] = table['average_temperature_monthly'].apply(lambda row: cleaning_dataframe(row))
    table.fillna(method='ffill', inplace=True)
    plot_temperature(table)
    resample_plot_temperature(table)
    
    
    """ CO2 emission from world bank """
    
    CO2_emission = pd.read_csv('data/API_EN.ATM.CO2E.PC_DS2_en_csv_v2_248248.csv', skiprows=3)
    CO2_emission.head()
    CO2_emission, table_CO2 = preprocess_CO2(CO2_emission)
    v = table_CO2.apply(lambda row: populate_CO2(row, CO2_emission), axis=1)
    table_CO2['Global CO2 Emissions per Capita'] = v
    table_CO2.set_index('date',inplace = True)
    table_CO2.fillna(method='ffill', inplace=True)

    plot_CO2(table_CO2)
    resample_plot_CO2(table_CO2)
    
    
    
    """ analysis the global tempurature with country and states """
    
    global_temp_country = pd.read_csv("data/GlobalLandTemperaturesByCountry.csv")
    print(type(global_temp_country))
    # select the country in below
    global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(
    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',
     'United Kingdom', 'Africa', 'South America'])]
    # rename the country
    global_temp_country_clear = global_temp_country_clear.replace(
       ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],
       ['Denmark', 'France', 'Netherlands', 'United Kingdom'])
    
    # get the mean temperature in those countries
    countries = np.unique(global_temp_country_clear['Country'])
    mean_temp = []
    for country in countries:
        mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == country]['AverageTemperature'].mean())
    
    # plot the global ball
    print_global_plot(global_temp_country,countries,mean_temp)
    Global_flat(global_temp_country,global_temp_country_clear,countries)
    
    
    """ analysis the global temperature """
    global_temp = pd.read_csv("data/GlobalTemperatures.csv")
    
    Global_temp(global_temp, global_temp_country_clear)
    average_tempurature_country(global_temp_country_clear)


def monthly_analysis():
    
def ARIMA_analysis():
    """ read in data"""
    global_temp = pd.read_csv("data/GlobalTemperatures.csv",index_col="dt",infer_datetime_format=True)
    global_temp.head()

    LandAndOceanAverageTemperature = global_temp.LandAndOceanAverageTemperature
    missing_dates = LandAndOceanAverageTemperature[LandAndOceanAverageTemperature.isnull() == True]
    print(missing_dates.tail())
    
    """ select recent data gte the mean and var then plot"""
    recent = global_temp.LandAndOceanAverageTemperature["1850":]
    recent.isnull().sum()
    var = recent.rolling(12).std()
    mean = recent.rolling(12).mean()
    mean.plot()
    plt.title("Mean of Global Average Temperature post 1850")
    plt.xlabel("Time")
    plt.ylabel("Average Temperature")
    var.plot()
    plt.title("Std of Global Average Temperature post 1850")
    plt.xlabel("Time")
    plt.ylabel("Average Temperature")
    # 以上并不stationary
    
    """if we drop the NaN data then plot """
    diff = recent.diff().dropna()
    mean_diff = diff.rolling(12).mean()
    var_diff = diff.rolling(12).std()
    diff.plot()
    mean_diff.plot(c = "red")
    var_diff.plot(c = "green")
    plt.xlabel("Time")
    # 做以下处理的原因是
    # 在ARMA/ARIMA这样的自回归模型中，模型对时间序列数据的平稳是有要求的，
    # 因此，需要对数据或者数据的n阶差分进行平稳检验，而一种常见的方法就是ADF检验，即单位根检验。
    dftest = sm.tsa.adfuller(diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    
    #ARIMA
    sm.tsa.graphics.plot_acf(diff,lags = np.arange(0,25,1))
    sm.tsa.graphics.plot_pacf(diff,lags=np.arange(0,25,1))
    mod = sm.tsa.SARIMAX(recent,order = (3,1,0), seasonal_order=(0,0,0,12)).fit()
    mod.summary()
    
    mod.plot_diagnostics()
    
    
def main():
    high_view()
    monthly_analysis()
    ARIMA_analysis()


main()