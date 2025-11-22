# Student Performance Predictor

A lightweight Streamlit app that predicts student exam performance from study habits and provides personalized advice. The project combines a pre-trained ML regressor (saved as model.pkl) with a rules-based recommendation engine and a context-aware sentiment feedback feature to support students.

## Key features
- Predicts expected grade (letter grade) from inputs such as study hours, attendance, sleep, GPA, and more.
- Rule-based, contextual advice based on predicted grade and user inputs.
- Study habit radar chart for a quick visual overview.
- Save predictions and update actual grades over time.
- Sentiment feedback analyzer (VADER/TextBlob) that uses the user's input + predicted grade to generate human-like encouraging responses.
- Simple user accounts (sign-up/login) with hashed passwords.

## Tech stack
- Python 3.9+
- Streamlit (app UI)
- scikit-learn, joblib (model & scaler)
- pandas, numpy (data handling)
- plotly (charts)
- vaderSentiment / textblob (sentiment analysis)
- SMTP (email contact form)

## Project structure (important files)
/src

  └─ app.py                # Main Streamlit app

  └─ train.py 

  └─ users.pkl             # Stored users (created at runtime)
  
/model.pkl                 # Trained ML model (RandomForestRegressor)

/scaler.pkl                # Preprocessing scaler

/data

  └─ predictions.csv      # Saved predictions (created at runtime)

  └─ StudentPerformanceFactors.csv

requirements.txt

Make sure model.pkl and scaler.pkl are in the project root (or update paths in app.py) before running.

## How to download & run (step-by-step)

### Clone the repo

`git clone https://github.com/Oshini-Sandunika/StudentPerformancePredictor.git`

`cd StudentPerformancePredictor`


### Create a Python virtual environment

`python -m venv venv`

#### Windows
`venv\Scripts\activate`
#### macOS / Linux
`source venv/bin/activate`


### Install dependencies

`pip install --upgrade pip`

`pip install -r requirements.txt`


If you don't have a requirements.txt, create it with the main libs:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- plotly
- textblob
- vaderSentiment


### Place model & scaler

Copy model.pkl and scaler.pkl into the repository root (same folder as app.py).

If you trained the model in another folder, update load_model_and_scaler() paths in app.py.

### Run the app

`streamlit run src/app.py`


Open the URL printed in the terminal (usually http://localhost:8501
).

### Stopping the app

Press Ctrl+C in the terminal running Streamlit.

If a stubborn python process remains, find the PID and kill it (taskkill /F /IM python.exe on Windows or kill -9 PID on macOS/Linux).

## How the workflow works (quick)

- Login / Sign up.

- Fill the prediction form (subject name, hours, attendance, etc.).

- Click Predict Grade → model predicts a score → converted to letter grade; advice and radar chart shown.

- You can write feedback in the feedback box.

- Feedback is analyzed using sentiment + context (your inputs + predicted grade) and returns a human-like encouraging message.

## Tips to improve further

- Store user data and models in a proper DB (SQLite / Postgres) instead of CSV for production.

- Include a server-side LLM API (OpenAI/Hugging Face Inference) for better conversational quality.

- Add authentication and session security for multi-user deployments.

- Add unit tests for the mapping function that converts predicted score → letter grade.

## Credits & Thanks

Thanks to my project teammates and friends for their help and support during development!

Repository: https://github.com/Oshini-Sandunika/StudentPerformancePredictor.git

