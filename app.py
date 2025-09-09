from flask import Flask, request, render_template
import pandas as pd
from pickle import load

# Load the model
with open("pm.pkl", "rb") as f:
    model = load(f)

# Load feature column order
with open("pm_cols.pkl", "rb") as f:
    cols = load(f)

app = Flask(__name__)

# Function for PM2.5 category
def pm25_category(value):
    if 0 <= value <= 50:
        return "Good"
    elif 51 <= value <= 100:
        return "Moderate"
    elif 101 <= value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= value <= 200:
        return "Unhealthy"
    elif 201 <= value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# Function for PM2.5 message
def pm25_message(category):
    messages = {
        "Good": "Air quality is safe. Enjoy your day outside! 🌞",
        "Moderate": "Air is acceptable, but sensitive people should take care. 🌤️",
        "Unhealthy for Sensitive Groups": "Limit outdoor activity if you have health issues. 😷",
        "Unhealthy": "Avoid outdoor activities, especially for children and elderly. 🚫",
        "Very Unhealthy": "Serious health effects possible. Stay indoors. 🏠",
        "Hazardous": "Emergency condition! Avoid all outdoor exposure. 🚨"
    }
    return messages.get(category, "")

@app.route("/", methods=["GET", "POST"])
def home():
    res = None
    category = None
    message = None
    city = None
    pm25 = None

    if request.method == "POST":
        city = request.form.get("city")
        pm25 = float(request.form.get("pm25"))

        # ----- Create DataFrame -------
        guide = {"city": city, "pm25": pm25}
        guide_df = pd.DataFrame([guide])

        # Encode categorical variables
        guide_df = pd.get_dummies(guide_df)
        guide_df = guide_df.reindex(columns=cols, fill_value=0)

        # ---- Predict ----
        res = model.predict(guide_df)[0]

        # ---- PM2.5 Category ----
        category = pm25_category(pm25)

        # ---- PM2.5 Message ----
        message = pm25_message(category)

    return render_template("home.html", 
                           res=res, 
                           city=city, 
                           pm25=pm25, 
                           category=category,
                           message=message)

if __name__ == "__main__":
    app.run(debug=True)
