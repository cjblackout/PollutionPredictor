import math
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from pickle import load
from keras.models import load_model
from dotenv import load_dotenv

def sin_cos(x, mode: str) -> float:
    denom = 2*math.pi
    if mode == 'hour':
        return math.sin((int(x)/24)*denom) + math.cos((int(x)/24)*denom)
    elif mode == 'day':
        return math.sin((int(x)/31)*denom) + math.cos((int(x)/31)*denom)
    else:
        return math.sin((int(x)/12)*denom) + math.cos((int(x)/12)*denom)

def deg2rad(deg) -> float:
    return math.sin(deg * (math.pi/180)) + math.cos(deg * (math.pi/180))

def get_aqi_label(df: pd.DataFrame) -> pd.Series:

    labels=["Good","Moderate","Unhealthy for Sensitive"," Unhealthy","Very Unhealthy", "Severe", "Hazardous"]
    bins = [0,50,100,150,200,300,400,500]
    aqi_labels = pd.cut(df['AQI'], bins=bins, labels=labels)

    return aqi_labels

def aqi_message(aqi_label: str) -> str:
    if aqi_label == "Good":
        return "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif aqi_label == "Moderate":
        return "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution."
    elif aqi_label == "UnhealthyforSensitiveGroups":
        return "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
    elif aqi_label == "Unhealthy":
        return "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_label == "VeryUnhealthy":
        return "Health alert: everyone may experience more serious health effects."
    elif aqi_label == "Severe":
        return "Health warnings of emergency conditions. The entire population is more likely to be affected."
    elif aqi_label == "Hazardous":
        return "This is considered a health emergency, and requires governmental action."
    else:
        return "No AQI label found."
    
def get_data() -> pd.DataFrame:

    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    lat = "28.6667"
    long = "77.2167"
    
    weather_url = "https://api.openweathermap.org/data/2.5/onecall?lat="+lat+"&lon="+long+"&appid="+str(API_KEY)
    pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution?lat="+lat+"&lon="+long+"&appid="+str(API_KEY)

    #get the response from the pollution_url and base_url 
    weather_response = requests.get(weather_url)
    weather = weather_response.json()

    pollution_reponse = requests.get(pollution_url)
    pollution = pollution_reponse.json()

    # Create a new dataframe with the input values
    datetime_temp = pd.DataFrame({
        "maxtempC": [int(weather['daily'][0]['temp']['max'])-273.15],
        "mintempC": [int(weather['daily'][0]['temp']['min'])-273.15],
        "sunHour": [(int(weather['daily'][0]['sunset']) - int(weather['daily'][0]['sunrise']))/3600],
        "uvindex.1": [int(weather['hourly'][0]['uvi'])],
        "sunrise": [datetime.fromtimestamp(int(weather['daily'][0]['sunrise'])).strftime("%H:%M:%S")],
        "sunset": [datetime.fromtimestamp(int(weather['daily'][0]['sunset'])).strftime("%H:%M:%S")],
        "DewPointC": [int(weather['hourly'][0]['dew_point'])-273.15],
        "FeelsLikeC": [int(weather['hourly'][0]['feels_like'])-273.15],
        "WindGustKmph": [int(weather['hourly'][0]['wind_gust'])],
        "cloudcover": [int(weather['hourly'][0]['clouds'])],
        "humidity": [int(weather['hourly'][0]['humidity'])],
        "precipMM": [int(weather['minutely'][0]['precipitation'])],
        "pressure": [int(weather['hourly'][0]['pressure'])],
        "tempC": [int(weather['hourly'][0]['temp'])-273.15],
        "visibility": [int(weather['hourly'][0]['visibility'])/1000],
        "winddirDegree": [int(weather['hourly'][0]['wind_deg'])],
        "windspeedKmph": [int(weather['hourly'][0]['wind_speed'])],
        "PM2.5": [pollution['list'][0]['components']['pm2_5']],
        "PM10": [pollution['list'][0]['components']['pm10']],
        "AQI": [pollution['list'][0]['main']['aqi']*45],
        "hour": [datetime.now().hour],
        "day": [datetime.now().day],
        "month": [datetime.now().month],
    })

    # Add 6 hour delayed values of the same dataframe
    new_datetime = datetime.now() + timedelta(hours=6)
    datetime_temp_6h = datetime_temp.copy()
    datetime_temp_6h["uvindex.1"] = int(weather['hourly'][5]['uvi'])
    datetime_temp_6h["DewPointC"] = int(weather['hourly'][5]['dew_point'])-273.15
    datetime_temp_6h["FeelsLikeC"] = int(weather['hourly'][5]['feels_like'])-273.15
    datetime_temp_6h["WindGustKmph"] = int(weather['hourly'][5]['wind_gust'])
    datetime_temp_6h["cloudcover"] = int(weather['hourly'][5]['clouds'])
    datetime_temp_6h["humidity"] = int(weather['hourly'][5]['humidity'])
    datetime_temp_6h["pressure"] = int(weather['hourly'][5]['pressure'])
    datetime_temp_6h["tempC"] = int(weather['hourly'][5]['temp'])-273.15
    datetime_temp_6h["visibility"] = int(weather['hourly'][5]['visibility'])/1000
    datetime_temp_6h["winddirDegree"] = int(weather['hourly'][5]['wind_deg'])
    datetime_temp_6h["windspeedKmph"] = int(weather['hourly'][5]['wind_speed'])
    datetime_temp_6h["hour"] = new_datetime.hour
    datetime_temp_6h["day"] = new_datetime.day
    datetime_temp_6h["month"] = new_datetime.month

    # Concatenate the two dataframes
    datetime_now = pd.concat([datetime_temp, datetime_temp_6h], ignore_index=True)

    #convert 'sunrise', 'sunset' to seconds
    datetime_now['sunrise'] = datetime_now['sunrise'].apply(lambda x: (int(x[:2])*3600 + int(x[3:5])*60)/86400)
    datetime_now['sunset'] = datetime_now['sunset'].apply(lambda x: (int(x[:2])*3600 + int(x[3:5])*60)/86400)

    #make columns 'hour', 'day', 'month', 'year' from Datetime
    datetime_now['hour'] = datetime_now['hour'].apply(lambda x: sin_cos(x,mode = 'hour'))
    datetime_now['day'] = datetime_now['day'].apply(lambda x: sin_cos(x, mode = 'day'))
    datetime_now['month'] = datetime_now['month'].apply(lambda x: sin_cos(x, mode = 'month'))

    #apply deg2rad to columns 'winddirDegree'
    datetime_now['winddirDegree'] = datetime_now['winddirDegree'].apply(deg2rad)

    return datetime_now

def predict_aqi(datetime_now: pd.DataFrame) -> str:
    scaler = load(open('scaler.pkl', 'rb'))
    model = load_model('model.h5')

    datetime_now = pd.DataFrame(scaler.transform(datetime_now), columns = datetime_now.columns)

    pred_AQI = model.predict(datetime_now) #type:ignore
    datetime_now.loc[0, "AQI"] = pred_AQI[0][0]
    datetime_now = pd.DataFrame(scaler.inverse_transform(datetime_now), columns = datetime_now.columns)

    AQI_labels = get_aqi_label(datetime_now)
    msg = "The AQI at this time, tomorrow, will be between " + str(round(datetime_now['AQI'][0])) + " and " + str(round(datetime_now['AQI'][1])) + " which is categorized as " + str(AQI_labels[0]) + ". " + str(aqi_message(AQI_labels[0].replace(" ","")))

    print(msg)
    return msg

if __name__ == '__main__':
    datetime_now = get_data()
    predict_aqi(datetime_now)