import os
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for
from flask_dance.contrib.fitbit import make_fitbit_blueprint, fitbit
from datetime import date

_ = load_dotenv()
app = Flask(__name__)
client_id = os.getenv("FITBIT_CLIENT_ID")
client_secret = os.getenv("FITBIT_CLIENT_SECRET")
app.secret_key = os.getenv("secret_key")

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

FITBIT_SCOPES = [
    "activity",
    # "heartrate",
    # "location",
    # "nutrition",
    "profile",
    # "settings",
    # "sleep",
    # "social",
    # "weight",
]

blueprint = make_fitbit_blueprint(
    client_id=client_id,
    client_secret=client_secret,
    redirect_url="http://127.0.0.1:5000/login/fitbit/authorized/",
    # reprompt_consent=True,
    scope=FITBIT_SCOPES,
)
app.register_blueprint(blueprint, url_prefix="/login")


@app.route("/")
def index():
    fitbit_data = None
    # user_info_endpoint = "/oauth2/v2/userinfo"
    # user_info_endpoint = "1/user/-/profile.json"
    # user_info_endpoint = "1/user/-/activities.json"
    user_info_endpoint = f"1/user/-/activities/date/{str(date.today())}.json"
    if fitbit.authorized:
        fitbit_data = fitbit.get(user_info_endpoint).json()["summary"]
        # print(fitbit_data["user"]["age"])

    return render_template(
        "index.j2", fitbit_data=fitbit_data, fetch_url=fitbit.base_url + user_info_endpoint
    )


@app.route("/login")
def login():
    return redirect(url_for("fitbit.login"))


if __name__ == "__main__":
    app.run(debug=True)

# https://github.com/lila/flask-dance-fitbit/blob/main/fitbit.py