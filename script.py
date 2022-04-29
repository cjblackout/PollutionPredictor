#import tensorflow, sklearn, matplotlib, numpy and pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import requests
import json
import pywhatkit

from datetime import timedelta
from matplotlib import pyplot
from pickle import load
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from datetime import datetime
from sklearn.metrics import mean_squared_error

#write a function that takes input in radians and returns sin + cos of that input
def sin_cos(x, mode):
    denom = 2*math.pi

    if mode == 'hour':
        return math.sin((int(x)/24)*denom) + math.cos((int(x)/24)*denom)
    elif mode == 'day':
        return math.sin((int(x)/31)*denom) + math.cos((int(x)/31)*denom)
    else:
        return math.sin((int(x)/12)*denom) + math.cos((int(x)/12)*denom)

#write a function that converts degrees to radians
def deg2rad(deg):
    return math.sin(deg * (math.pi/180)) + math.cos(deg * (math.pi/180))

def get_IQA_label(df):
    labels=["Good","Moderate","Unhealthy for Sensitive"," Unhealthy","Very Unhealthy", "Severe", "Hazardous"]
    bins = [0,50,100,150,200,300,400,500]
    #bin the values of df['AQI'] into bins of size 50 each and assign the labels to the bins
    AQI_labels = pd.cut(df['AQI'], bins=bins, labels=labels)
    return AQI_labels

#create a function that displays a message for each AQI label
def AQI_message(AQI_label):
    if AQI_label == "Good":
        return "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif AQI_label == "Moderate":
        return "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution."
    elif AQI_label == "UnhealthyforSensitiveGroups":
        return "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
    elif AQI_label == "Unhealthy":
        return "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    elif AQI_label == "VeryUnhealthy":
        return "Health alert: everyone may experience more serious health effects."
    elif AQI_label == "Severe":
        return "Health warnings of emergency conditions. The entire population is more likely to be affected."
    elif AQI_label == "Hazardous":
        return "This is considered a health emergency, and requires governmental action."

#load scaler.pkl file
scaler = load(open('personal\Pollution\scaler.pkl', 'rb'))
model = load_model('personal\Pollution\model.h5')

#configure pandas to show the full dataframe
pd.set_option('display.max_columns', None)

# Enter your API key here
api_key = "c36b466e52b1c3d2e56d88cde5xxxxxx"

# Give city location
lat = "28.6667"
long = "77.2167"
 
#make pollution_url base_url variable to store url
base_url = "https://api.openweathermap.org/data/2.5/onecall?lat="+lat+"&lon="+long+"&appid="+api_key
pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution?lat="+lat+"&lon="+long+"&appid="+api_key
 
#get the response from the pollution_url and base_url 
response = requests.get(base_url)
weather = response.json()
pollution = requests.get(pollution_url)
pollution = pollution.json()

#make a new dataframe with index
datetime_temp = pd.DataFrame(index = [0])
datetime_now = pd.DataFrame()

#add the values from the JSON to the dataframe
datetime_temp['maxtempC'] = int(weather['daily'][0]['temp']['max'])-273.15
datetime_temp['mintempC'] = int(weather['daily'][0]['temp']['min'])-273.15
datetime_temp['sunHour'] = (int(weather['daily'][0]['sunset']) - int(weather['daily'][0]['sunrise']))/3600
datetime_temp['uvindex.1'] = int(weather['hourly'][0]['uvi'])
datetime_temp['sunrise'] = datetime.fromtimestamp(int(weather['daily'][0]['sunrise'])).strftime("%H:%M:%S")
datetime_temp['sunset'] = datetime.fromtimestamp(int(weather['daily'][0]['sunset'])).strftime("%H:%M:%S")
datetime_temp['DewPointC'] = int(weather['hourly'][0]['dew_point'])-273.15
datetime_temp['FeelsLikeC'] = int(weather['hourly'][0]['feels_like'])-273.15
datetime_temp['WindGustKmph'] = int(weather['hourly'][0]['wind_gust'])
datetime_temp['cloudcover'] = int(weather['hourly'][0]['clouds'])
datetime_temp['humidity'] = int(weather['hourly'][0]['humidity'])
datetime_temp['precipMM'] = int(weather['minutely'][0]['precipitation'])
datetime_temp['pressure'] = int(weather['hourly'][0]['pressure'])
datetime_temp['tempC'] = int(weather['hourly'][0]['temp'])-273.15
datetime_temp['visibility'] = int(weather['hourly'][0]['visibility'])/1000
datetime_temp['winddirDegree'] = int(weather['hourly'][0]['wind_deg'])
datetime_temp['windspeedKmph'] = int(weather['hourly'][0]['wind_speed'])
datetime_temp['PM2.5'] = pollution['list'][0]['components']['pm2_5']
datetime_temp['PM10'] = pollution['list'][0]['components']['pm10']
datetime_temp['AQI'] = pollution['list'][0]['main']['aqi']*45
datetime_temp['hour'] = int(datetime.now().strftime("%H"))
datetime_temp['day'] = int(datetime.now().strftime("%d"))
datetime_temp['month'] = int(datetime.now().strftime("%m"))

#add datetime_temp to datetime_now
datetime_now = datetime_now.append(datetime_temp, ignore_index=True)
datetime_temp = pd.DataFrame(index = [1])
new_datetime = datetime.now() + timedelta(hours=6)

#Add 6 hour delayed values of the same dataframe
datetime_temp['maxtempC'] = int(weather['daily'][0]['temp']['max'])-273.15
datetime_temp['mintempC'] = int(weather['daily'][0]['temp']['min'])-273.15
datetime_temp['sunHour'] = (int(weather['daily'][0]['sunset']) - int(weather['daily'][0]['sunrise']))/3600
datetime_temp['uvindex.1'] = int(weather['hourly'][5]['uvi'])
datetime_temp['sunrise'] = datetime.fromtimestamp(int(weather['daily'][0]['sunrise'])).strftime("%H:%M:%S")
datetime_temp['sunset'] = datetime.fromtimestamp(int(weather['daily'][0]['sunset'])).strftime("%H:%M:%S")
datetime_temp['DewPointC'] = int(weather['hourly'][5]['dew_point'])-273.15
datetime_temp['FeelsLikeC'] = int(weather['hourly'][5]['feels_like'])-273.15
datetime_temp['WindGustKmph'] = int(weather['hourly'][5]['wind_gust'])
datetime_temp['cloudcover'] = int(weather['hourly'][5]['clouds'])
datetime_temp['humidity'] = int(weather['hourly'][5]['humidity'])
datetime_temp['precipMM'] = int(weather['minutely'][0]['precipitation'])
datetime_temp['pressure'] = int(weather['hourly'][5]['pressure'])
datetime_temp['tempC'] = int(weather['hourly'][5]['temp'])-273.15
datetime_temp['visibility'] = int(weather['hourly'][5]['visibility'])/1000
datetime_temp['winddirDegree'] = int(weather['hourly'][5]['wind_deg'])
datetime_temp['windspeedKmph'] = int(weather['hourly'][5]['wind_speed'])
datetime_temp['PM2.5'] = pollution['list'][0]['components']['pm2_5']
datetime_temp['PM10'] = pollution['list'][0]['components']['pm10']
datetime_temp['AQI'] = pollution['list'][0]['main']['aqi']*45
datetime_temp['hour'] = int(new_datetime.strftime("%H"))
datetime_temp['day'] = int(new_datetime.strftime("%d"))
datetime_temp['month'] = int(new_datetime.strftime("%m"))

datetime_now = datetime_now.append(datetime_temp, ignore_index=True)

#convert 'sunrise', 'sunset' to seconds
datetime_now['sunrise'] = datetime_now['sunrise'].apply(lambda x: (int(x[:2])*3600 + int(x[3:5])*60)/86400)
datetime_now['sunset'] = datetime_now['sunset'].apply(lambda x: (int(x[:2])*3600 + int(x[3:5])*60)/86400)

#make columns 'hour', 'day', 'month', 'year' from Datetime
datetime_now['hour'] = datetime_now['hour'].apply(lambda x: sin_cos(x,mode = 'hour'))
datetime_now['day'] = datetime_now['day'].apply(lambda x: sin_cos(x, mode = 'day'))
datetime_now['month'] = datetime_now['month'].apply(lambda x: sin_cos(x, mode = 'month'))

#apply deg2rad to columns 'winddirDegree'
datetime_now['winddirDegree'] = datetime_now['winddirDegree'].apply(deg2rad)

datetime_now = pd.DataFrame(scaler.transform(datetime_now), columns = datetime_now.columns)

pred_AQI = model.predict(datetime_now)
datetime_now.loc[0, "AQI"] = pred_AQI[0][0]
datetime_now = pd.DataFrame(scaler.inverse_transform(datetime_now), columns = datetime_now.columns)

AQI_labels = get_IQA_label(datetime_now)
msg = "This is an automated message. The AQI at this time, tomorrow, will be between " + str(round(datetime_now['AQI'][0])) + " and " + str(round(datetime_now['AQI'][1])) + " which is categorized as " + str(AQI_labels[0]) + ". " + str(AQI_message(AQI_labels[0].replace(" ",""))) + " Greetings from Rishabh Saxena"
number = input("Enter the contact number: ")
pywhatkit.sendwhatmsg_instantly(str(number), msg)