from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('AdaBoostRegressor.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        season = request.form['season']
        if (season == 'Spring'):
            season_Fall = 0
            season_Spring = 1
            season_Summer = 0
            season_Winter = 0
        elif (season == 'Summer'):
            season_Fall = 0
            season_Spring = 0
            season_Summer = 1
            season_Winter = 0
        elif (season == 'Fall'):
            season_Fall = 1
            season_Spring = 0
            season_Summer = 0
            season_Winter = 0
        else :
            season_Fall = 0
            season_Spring = 0
            season_Summer = 0
            season_Winter = 1
        month=int(request.form['month'])
        hour=int(request.form['hour'])
        holiday = int(request.form['holiday'])

        weekday = request.form['weekday']
        if (weekday == 0):
            weekday_sunday = 1
            weekday_monday = 0
            weekday_tuesday = 0
            weekday_wednesday = 0
            weekday_thursday = 0
            weekday_friday = 0
            weekday_saturday = 0
        elif (weekday == 1):
            weekday_sunday = 0
            weekday_monday = 1
            weekday_tuesday = 0
            weekday_wednesday = 0
            weekday_thursday = 0
            weekday_friday = 0
            weekday_saturday = 0
        elif (weekday == 2):
            weekday_sunday = 0
            weekday_monday = 0
            weekday_tuesday = 1
            weekday_wednesday = 0
            weekday_thursday = 0
            weekday_friday = 0
            weekday_saturday = 0
        elif (weekday == 3):
            weekday_sunday = 0
            weekday_monday = 0
            weekday_tuesday = 0
            weekday_wednesday = 1
            weekday_thursday = 0
            weekday_friday = 0
            weekday_saturday = 0
        elif (weekday == 4):
            weekday_sunday = 0
            weekday_monday = 0
            weekday_tuesday = 0
            weekday_wednesday = 0
            weekday_thursday = 1
            weekday_friday = 0
            weekday_saturday = 0
        elif (weekday == 5):
            weekday_sunday = 0
            weekday_monday = 0
            weekday_tuesday = 0
            weekday_wednesday = 0
            weekday_thursday = 0
            weekday_friday = 1
            weekday_saturday = 0
        else:
            weekday_sunday = 0
            weekday_monday = 0
            weekday_tuesday = 0
            weekday_wednesday = 0
            weekday_thursday = 0
            weekday_friday = 0
            weekday_saturday = 1
        workingday = int(request.form['workingday'])
        weather = int(request.form['weather'])
        if (weather==0):
            weather_Clear = 1
            weather_Heavy_Snow_Rain = 0
            weather_Light_Snow_Rain = 0
            weather_Misty_Cloudy = 0
        elif (weather == 1):
            weather_Clear = 0
            weather_Heavy_Snow_Rain = 1
            weather_Light_Snow_Rain = 0
            weather_Misty_Cloudy = 0
        elif (weather == 2):
            weather_Clear = 0
            weather_Heavy_Snow_Rain = 0
            weather_Light_Snow_Rain = 1
            weather_Misty_Cloudy = 0
        else:
            weather_Clear = 0
            weather_Heavy_Snow_Rain = 0
            weather_Light_Snow_Rain = 0
            weather_Misty_Cloudy = 1
        temp = float(request.form['temp'])
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        casual = int(request.form['casual'])
        registered = int(request.form['registered'])

        prediction=model.predict([[month,hour,holiday,workingday,temp,humidity,windspeed,casual,registered,season_Fall,season_Spring
                                      ,season_Summer,season_Winter,weekday_friday,weekday_monday,weekday_saturday,weekday_sunday
                                   ,weekday_thursday,weekday_tuesday,weekday_wednesday,weather_Clear,weather_Heavy_Snow_Rain
                                   ,weather_Light_Snow_Rain,weather_Misty_Cloudy]])
        output=round(np.exp(prediction[0]))

        return render_template('index.html',prediction_text="The bike sharing  count is  : {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

