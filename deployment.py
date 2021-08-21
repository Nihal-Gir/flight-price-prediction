''' Flight Price Prediction '''

from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("D:/Studies/Course Material/flight_price_prediction/random_forest_regressor.pkl","rb"))

@app.route("/", methods = ['GET'])
def home():
    return render_template("price.html")

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':

        departure = request.form['departure']
        Journey_day = int(pd.to_datetime(departure, format = '%Y-%m-%dT%H:%M').day)
        Journey_month = int(pd.to_datetime(departure, format = '%Y-%m-%dT%H:%M').month)
        Dep_hour = int(pd.to_datetime(departure, format = '%Y-%m-%dT%H:%M').hour)
        Dep_min = int(pd.to_datetime(departure, format = '%Y-%m-%dT%H:%M').minute)

        arrival = request.form['arrival']
        Arrival_hour = int(pd.to_datetime(arrival, format = '%Y-%m-%dT%H:%M').hour)
        Arrival_min = int(pd.to_datetime(arrival, format = '%Y-%m-%dT%H:%M').minute)

        Duration_hours = abs(Arrival_hour - Dep_hour)
        Duration_mins = abs(Arrival_min - Dep_min)

        stops = request.form['stops']
        if stops == 'non-stop':
            Total_Stops = 0
        elif stops == '1 stop':
            Total_Stops = 1
        elif stops == '2 stops':
            Total_Stops = 2    
        elif stops == '3 stops':
            Total_Stops = 3    
        else:
            Total_Stops = 4

        source = request.form['source']
        if source == 'banglore':
            Source_Banglore = 1
            Source_Chennai = 0
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
        elif source == 'chennai':
            Source_Banglore = 0
            Source_Chennai = 1
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
        elif source == 'delhi':
            Source_Banglore = 0
            Source_Chennai = 0
            Source_Delhi = 1
            Source_Kolkata = 0
            Source_Mumbai = 0
        elif source == 'kolkata':
            Source_Banglore = 0
            Source_Chennai = 0
            Source_Delhi = 0
            Source_Kolkata = 1
            Source_Mumbai = 0
        else:
            Source_Banglore = 0
            Source_Chennai = 0
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 1

        destination = request.form['destination']
        if destination == 'banglore':
            Destination_Banglore = 1
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_NewDelhi = 0
        elif destination == 'cochin':
            Destination_Banglore = 1
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_NewDelhi = 0
        elif destination == 'delhi':
            Destination_Banglore = 0
            Destination_Cochin = 0
            Destination_Delhi = 1
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_NewDelhi = 0
        elif destination == 'kolkata':
            Destination_Banglore = 0
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 1
            Destination_NewDelhi = 0
        elif destination == 'hyderabad':
            Destination_Banglore = 0
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 1
            Destination_Kolkata = 0
            Destination_NewDelhi = 0
        else:
            Destination_Banglore = 0
            Destination_Cochin = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_NewDelhi = 1

        airline = request.form['airline']
        if airline == 'air asia':
            AirAsia = 1
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'air india':
            AirAsia = 0
            AirIndia = 1
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'go air':
            AirAsia = 0
            AirIndia = 0
            GoAir = 1
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'indigo':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 1
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'jet airways':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 1
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'jet airways business':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 1
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'multiple carriers':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 1
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'multiple carriers premium economy':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 1
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'spicejet':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 1
            Trujet = 0
            GoAir = 0
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'trujet':
            AirAsia = 0
            AirIndia = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 1
            Vistara = 0
            VistaraPremiumeconomy = 0
        elif airline == 'vistara':
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 1
            VistaraPremiumeconomy = 0
        else:
            AirAsia = 0
            AirIndia = 0
            GoAir = 0
            IndiGo = 0
            JetAirways = 0
            JetAirwaysBusiness = 0
            Multiplecarriers = 0
            MultiplecarriersPremiumeconomy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            VistaraPremiumeconomy = 1
        
        prediction = model.predict([[Total_Stops,Journey_day,Journey_month,Dep_hour,Dep_min,Arrival_hour,Arrival_min,
                                     Duration_hours,Duration_mins,AirAsia,AirIndia,GoAir,IndiGo,JetAirways,JetAirwaysBusiness,
                                     Multiplecarriers,MultiplecarriersPremiumeconomy,SpiceJet,Trujet,Vistara,VistaraPremiumeconomy,
                                     Source_Banglore,Source_Chennai,Source_Delhi,Source_Kolkata,Source_Mumbai,Destination_Banglore,
                                     Destination_Cochin,Destination_Delhi,Destination_Hyderabad,
                                     Destination_Kolkata,Destination_NewDelhi]])
        output = round(prediction[0],2)

        return render_template("price.html", price_predictions = 'The price of your flight ticket is Rs. {}'.format(output))
    
    else:
        return render_template("price.html")

if __name__ == '__main__':
    app.run(debug = True)


