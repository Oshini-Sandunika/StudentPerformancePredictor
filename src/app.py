import streamlit as st
import joblib
import numpy as np
import os
import pickle
import hashlib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json, hashlib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random



# ---------------- HELPER FUNCTIONS ----------------
def encode(value, options):
    return options.index(value)

def make_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists("users.pkl"):
        with open("users.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def save_users(users):
    with open("users.pkl", "wb") as f:
        pickle.dump(users, f)

def load_model_and_scaler():
    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "model.pkl")
        scaler_path = os.path.join(project_dir, "scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        st.error("⚠️ Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in the main project folder.")
        st.stop()

def save_prediction(username, input_data, predicted_grade):
    # --- Ensure the data folder exists ---
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_folder, exist_ok=True)
    data_file = os.path.join(data_folder, "predictions.csv")

    # --- Build the record ---
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Username": username,
        "subject_name": input_data.get("subject_name", ""),  # added subject
        "Hours_Studied": input_data["hours"],
        "Attendance": input_data["attendance"],
        "Sleep_Hours": input_data["sleep_hours"],
        "GPA": input_data["gpa"],
        "Mid_Exam_Grade": input_data["grade"],
        "Motivation": input_data["motivation"],
        "Internet_Access": input_data["internet_access"],
        "Tutoring_Sessions": input_data["tutoring_sessions"],
        "Physical_Activity": input_data["physical_activity"],
        "Predicted_Grade": predicted_grade,
        "Actual_Grade": ""  # blank for now
    }

    # --- Write or append ---
    if not os.path.exists(data_file):
        # Create new CSV with headers
        pd.DataFrame([record]).to_csv(data_file, index=False)
    else:
        # Append new row
        pd.DataFrame([record]).to_csv(data_file, mode="a", header=False, index=False)



def send_email(sender_email, sender_password, receiver_email, name, email, message):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"New Feedback from {name}"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        html = f"""
        <html>
            <body>
                <h3>New Feedback from AskWise Contact Form</h3>
                <p><b>Name:</b> {name}</p>
                <p><b>Email:</b> {email}</p>
                <p><b>Message:</b><br>{message}</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(html, "html"))

        # Connect to Gmail SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False


def make_sig(user, inputs_dict, grade):
    payload = json.dumps({"user": user, "inputs": inputs_dict, "grade": grade}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()








# ---------------- INITIALIZE PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "username" not in st.session_state:
    st.session_state.username = None

users = load_users()






# ---------------- LANDING PAGE ----------------
def show_landing_page():
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body { margin: 0; padding: 0; }
            .landing-container {
                text-align: center; 
                font-family: 'Montserrat', sans-serif; 
                margin-top: 20vh;
            }
            .landing-container h1 {
                font-size: 2.5rem; 
                font-weight: 700; 
                color: white; 
                margin-bottom: 1rem; 
            }
            .landing-container p {
                font-size: 1.3rem;
                color: #4b5563;
                margin-bottom: 3rem;
            }
            .stButton>button {
                background-color: transparent; 
                color: white; 
                font-size: 1.5rem; 
                padding: 0.8rem 2rem; 
                border-radius: 12px; 
                border: 0.5px solid white; 
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # Main content
    st.markdown("""
        <div class="landing-container">
            <h1>Student Performance Predictor</h1>
            <p>Discover how your study habits impact success — get personalized insights!</p>
        </div>
    """, unsafe_allow_html=True)

    # Streamlit buttons for handling navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Prediction"):
            if st.session_state.get("username"):
                st.session_state.page = "form"
            else:
                st.session_state.page = "login"
            st.rerun()
    with col1:
        if st.button("Model Details"):
            st.session_state.page = "model_details"
            st.rerun()
    with col3:
        if st.button("Contact Us"):
            st.session_state.page = "contact_us"
            st.rerun()








# ---------------- MODEL DETAILS PAGE ----------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def show_model_details_page():
    st.markdown("""
        <style>
            /* Hide landing page top-right buttons */
            .top-right-links { 
                display: none !important; 
            }
                
            /* Adjust heading sizes */
            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.2rem !important; }

            /* Table row index styling */
            table.dataframe th.row_heading { 
                text-align: center; 
            }
        </style>

        <div class="top-right-home" style="position: fixed; top: 20px; right: 30px; z-index: 100;">
            <button onclick="window.location.reload();">🏠 Back to Home</button>
        </div>
    """, unsafe_allow_html=True)

    st.title("Model Details - Student Performance Predictor")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "model.pkl")
    scaler_path = os.path.join(project_dir, "scaler.pkl")
    data_path = os.path.join(project_dir, "data", "StudentPerformanceFactors.csv")

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(data_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        df = pd.read_csv(data_path)

        st.subheader("✅ Model Info")
        st.write(f"**Model:** Random Forest Regressor")
        st.write(f"**Number of Estimators:** {model.n_estimators}")
        st.write(f"**Max Depth:** {model.max_depth}")
        st.write(f"**Min Samples Split:** {model.min_samples_split}")
        st.write(f"**Min Samples Leaf:** {model.min_samples_leaf}")
        st.write(f"**Max Features:** {model.max_features}")
        st.write(f"**Bootstrap:** {model.bootstrap}")

        # Feature Importance with row numbers starting at 1
        features = [
            "Hours_Studied",
            "Attendance",
            "Sleep_Hours",
            "Previous_Scores",
            "Motivation_Level",
            "Internet_Access",
            "Tutoring_Sessions",
            "Physical_Activity",
            "Study_Motivation",
            "Attendance_Effort"
        ]
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        fi_df.index += 1  # row numbers start from 1
        st.subheader("🌟 Feature Importances")
        st.table(fi_df)

        # Prepare test set metrics
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        df["Study_Motivation"] = df["Hours_Studied"] * df["Motivation_Level"]
        df["Attendance_Effort"] = df["Attendance"] * df["Tutoring_Sessions"]

        target_col = "Exam_Score"
        feature_cols = features
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test[feature_cols] = scaler.transform(X_test[feature_cols])
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Test Set Metrics")
        st.write(f"**R² Score:** {r2:.3f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

        st.subheader("🔍 Sample Predictions")
        sample_df = X_test.copy()
        sample_df["Actual"] = y_test
        sample_df["Predicted"] = y_pred
        sample_df = sample_df.reset_index(drop=True)
        sample_df.index += 1  # row numbers start from 1
        st.dataframe(sample_df.head(10))

    else:
        st.error("❌ Model, scaler, or dataset not found. Train the model first.")

    # Streamlit Back to Home button (for Python-side page switch)
    if st.button("Back to Home"):
        st.session_state.page = "landing"
        st.rerun()





# ---------------- CONTACT US PAGE ----------------
def show_contact_page():
    st.markdown("""
        <style>
            h2 {
                font-size: 1.8rem !important;
            }
            h3, .stSubheader {
                font-size: 1.2rem !important;
            }
            p, li {
                font-size: 0.95rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown("<h2>📬 Contact Us</h2>", unsafe_allow_html=True)
    st.markdown("<p font-size:15px;'>We’d love to hear from you! Feel free to share your thoughts, feedback, or collaboration ideas.</p>", unsafe_allow_html=True)

    # --- Team Members ---
    st.markdown("<h4>👩‍💻 Project Contributors</h4>", unsafe_allow_html=True)
    team = ["Oshini Sandunika", "Member 2", "Member 3"]
    for member in team:
        st.markdown(f"- **{member}**")

    # --- Contact Details ---
    st.markdown("<h4>📞 Get in Touch</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Email:** [oshinisandunika416@gmail.com](mailto:oshinisandunika416@gmail.com)  
    - **GitHub:** [Project Repository](https://github.com/your-repo)  
    - **LinkedIn:** [Team Profile](https://linkedin.com)
    """)

    # --- Feedback Form ---
    st.markdown("<h4>💬 Send us a Message</h4>", unsafe_allow_html=True)
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send Message")

        if submitted:
            if name and email and message:
                sender_email = "pixelpair2.0@gmail.com"  
                sender_password = "gfwo xrjc mmtc qfds"  
                receiver_email = "oshinisandunika416@gmail.com"

                if send_email(sender_email, sender_password, receiver_email, name, email, message):
                    st.success("✅ Message sent successfully!")
            else:
                st.warning("Please fill in all fields before sending.")

    # --- Back to Home Button ---
    if st.button("Back to Home"):
        st.session_state.page = "landing"
        st.rerun()









# ---------------- LOGIN PAGE ----------------
def show_login_page():
    st.title("🔐 Login to Continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    msg = ""
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    with col2:
        if st.button("Login"):
            if username in users and users[username] == make_hash(password):
                st.session_state.username = username
                st.session_state.page = "form"
                st.rerun()
            else:
                msg = "Invalid username or password!"              

    with col3:
        if st.button("Sign Up"):
            st.session_state.page = "signup"
            st.rerun()

    with col4:
        if st.button("Back to Home"):
            st.session_state.page = "landing"
            st.rerun()

    if(msg != ""):
        st.error(msg)







# ---------------- SIGN-UP PAGE ----------------
def show_signup_page():
    st.title("📝 Create New Account")
    new_username = st.text_input("Choose a Username")
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_username in users:
            st.warning("⚠️ Username already exists!")
        elif new_password != confirm_password:
            st.error("❌ Passwords do not match!")
        elif len(new_username.strip()) == 0 or len(new_password.strip()) == 0:
            st.warning("Please fill all fields.")
        else:
            users[new_username] = make_hash(new_password)
            save_users(users)
            st.success("✅ Account created successfully! You can now log in.")
            st.session_state.page = "login"
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()







# ---------------- FORM PAGE ----------------
def show_form_page():
    if "show_subject_warning" not in st.session_state:
        st.session_state.show_subject_warning = False
    if "predicted_grade" not in st.session_state:
        st.session_state.predicted_grade = None
    if "predicted_score" not in st.session_state:
        st.session_state.predicted_score = None
    if "saved_inputs" not in st.session_state:
        st.session_state.saved_inputs = None
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = ""
    if "analysis_msg" not in st.session_state:
        st.session_state.analysis_msg = None
    if "last_saved_sig" not in st.session_state:
        st.session_state.last_saved_sig = None

    def clear_prediction():
        """Reset results when any input changes."""
        st.session_state.predicted_grade = None
        st.session_state.analysis_msg = None
        st.session_state.saved_inputs = None
        st.session_state.user_feedback = ""


    if "show_subject_warning" not in st.session_state:
        st.session_state.show_subject_warning = False

    st.markdown("""
        <style>
            .form-container {
                background: rgba(255, 255, 255, 0.8);  
                padding: 40px;
                border-radius: 20px;
                max-width: 800px;
                margin: 40px auto;  
            }
            .stButton>button {
                background-color: transparent; 
                color: white; 
                font-size: 1.5rem; 
                padding: 0.8rem 2rem; 
                border-radius: 12px; 
                border: 0.5px solid white; 
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
            }
            h1 { text-align: center; color: #023e8a; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1>Predict Your Expected Grade</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:gray;'>Welcome, <b>{st.session_state.username}</b> 👋 Fill out the details below.</p>", unsafe_allow_html=True)

    with st.spinner("Loading form... Please wait ⏳"):
        # Load model and scaler
        model, scaler = load_model_and_scaler()

        # Input Fields
        subject_name = st.text_input("Subject Name", placeholder="Enter subject name (e.g., Database Systems)", on_change=clear_prediction)
        hours = st.number_input("Hours Studied per Week", min_value=0, max_value=100, value=10, on_change=clear_prediction)
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80, on_change=clear_prediction)
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=0, max_value=24, value=7, on_change=clear_prediction)
        gpa = st.number_input("Most Recent GPA (out of 4.0)", min_value=0.0, max_value=4.0, value=3.0, step=0.01, on_change=clear_prediction)
        grade = st.selectbox("Mid Exam Grade", ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"], on_change=clear_prediction)

        def grade_to_percentage(grade):
            mapping = {
                "A+": 92.5, "A": 77, "A-": 67, "B+": 62, "B": 57, "B-": 52,
                "C+": 47, "C": 42, "C-": 37, "D+": 32, "D": 27, "F": 12
            }
            return mapping.get(grade, 50)

        def previous_score_from_gpa_grade(gpa, grade):
            grade_weights = {
                "A+": 0.6, "A": 0.5, "A-": 0.4,
                "B+": 0.3, "B": 0.3, "B-": 0.3,
                "C+": 0.2, "C": 0.2, "C-": 0.2,
                "D+": 0.1, "D": 0.1, "F": 0.0
            }
            w_gpa = grade_weights.get(grade, 0.3)
            w_grade = 1 - w_gpa
            gpa_percentage = gpa * 25
            grade_percentage = grade_to_percentage(grade)
            return w_gpa * gpa_percentage + w_grade * grade_percentage

        previous_scores = previous_score_from_gpa_grade(gpa, grade)

        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], on_change=clear_prediction)
        internet_access = st.selectbox("Internet Access", ["No", "Yes"], on_change=clear_prediction)
        tutoring_sessions = st.number_input("Tutoring Sessions per Week", min_value=0, max_value=20, value=2, on_change=clear_prediction)
        physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=50, value=3, on_change=clear_prediction)

        study_motivation = hours * encode(motivation, ["Low", "Medium", "High"])
        attendance_effort = attendance * tutoring_sessions

        input_data = np.array([[  
            hours, attendance, sleep_hours, previous_scores,
            encode(motivation, ["Low", "Medium", "High"]),
            encode(internet_access, ["No", "Yes"]),
            tutoring_sessions, physical_activity, study_motivation, attendance_effort
        ]])

        # --- Scale input ---
        input_scaled = scaler.transform(input_data)

        # --- Prediction and display ---
        col1, col2, col3 = st.columns([1, 1, 1])
        predicted_grade = None

        with col1:
            if st.button("Predict Grade") or st.session_state.predicted_grade is not None:
                if subject_name.strip() == "":
                    st.session_state.show_subject_warning = True
                else:
                    predicted_score = model.predict(input_scaled)[0]

                    # Map predicted score to letter grade with +/- variations
                    if predicted_score >= 85:
                        predicted_grade = "A+"
                    elif predicted_score >= 70:
                        predicted_grade = "A"
                    elif predicted_score >= 65:
                        predicted_grade = "A-"
                    elif predicted_score >= 60:
                        predicted_grade = "B+"
                    elif predicted_score >= 55:
                        predicted_grade = "B"
                    elif predicted_score >= 50:
                        predicted_grade = "B-"
                    elif predicted_score >= 45:
                        predicted_grade = "C+"
                    elif predicted_score >= 40:
                        predicted_grade = "C"
                    elif predicted_score >= 35:
                        predicted_grade = "C-"
                    elif predicted_score >= 30:
                        predicted_grade = "D+"
                    elif predicted_score >= 25:
                        predicted_grade = "D"    
                    else:
                        predicted_grade = "F"

                    # persist in session_state so it survives the next rerun
                    st.session_state.predicted_grade = predicted_grade
                    st.session_state.predicted_score = float(predicted_score)

                    # --- Save to CSV ---
                    st.session_state.saved_inputs = {
                        "subject_name": subject_name,
                        "hours": hours,
                        "attendance": attendance,
                        "sleep_hours": sleep_hours,
                        "gpa": gpa,
                        "grade": grade,
                        "motivation": motivation,
                        "internet_access": internet_access,
                        "tutoring_sessions": tutoring_sessions,
                        "physical_activity": physical_activity
                    }

                    # Save to CSV exactly once per distinct prediction
                    sig = make_sig(st.session_state.username, st.session_state.saved_inputs, predicted_grade)
                    if st.session_state.last_saved_sig != sig:
                        save_prediction(st.session_state.username, st.session_state.saved_inputs, predicted_grade)
                        st.session_state.last_saved_sig = sig

                        
        # Display full-width warning outside columns
        if subject_name.strip() == "" and st.session_state.get("show_subject_warning", False):
            st.warning("⚠️ Please enter a subject name before predicting.")

        with col2:
            if st.button("Update Actual Grade"):
                st.session_state.page = "update_grade"
                st.rerun()

        with col3:
        # Logout Button
            if st.button("Logout"):
                st.session_state.username = None
                st.session_state.page = "landing"
                st.rerun()

        # --- Display result ---
        if predicted_grade is not None:
            st.markdown("---")
            st.subheader(f"📝 **Predicted Grade:** {st.session_state.predicted_grade}")

            # pull frozen inputs from the time of prediction (prevents re-run drift)
            si = st.session_state.saved_inputs
            hours = si["hours"]
            attendance = si["attendance"]
            sleep_hours = si["sleep_hours"]
            motivation = si["motivation"]
            tutoring_sessions = si["tutoring_sessions"]
            physical_activity = si["physical_activity"]

            # ---------- RADAR CHART ----------
            st.markdown("<h5 style='text-align:center; color:white;'>Study Habit Radar Analysis</h5>", unsafe_allow_html=True)

            # Define categories (the habits)
            categories = ["Study Hours", "Attendance", "Sleep Hours", "Motivation", 
                  "Tutoring Sessions", "Physical Activity"]

            # Convert motivation to numeric for the radar
            motivation_score = encode(motivation, ["Low", "Medium", "High"]) * 50  # 0, 50, 100 scale

            # Normalize numeric values to roughly 0–100 scale for fair visualization
            values = [
                min(hours, 100),                          # Study hours
                attendance,                               # Attendance %
                sleep_hours * (100/10),                   # Assuming 10h = 100%
                motivation_score,                         # Encoded motivation
                tutoring_sessions * (100/10),             # Assuming 10 sessions = 100%
                physical_activity * (100/10)              # Assuming 10h activity = 100%
            ]

            # Close the radar chart loop
            values += values[:1]
            categories += categories[:1]

            # Create radar chart
            fig = go.Figure(
                data=[
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name='Your Study Profile',
                        line_color='deepskyblue',
                        fillcolor='rgba(0, 191, 255, 0.3)'
                    )
                ],
                layout=go.Layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    showlegend=False,
                    template="plotly_white",
                    margin=dict(t=20, b=20)
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("This chart shows how balanced your current habits are across key areas that influence performance.")

            # ---------- END RADAR CHART ----------

            # --- RULE-BASED ADVICE ---
            advice_list = []

            # Define context flags
            high_effort = hours >= 25
            moderate_effort = 15 <= hours < 25
            low_effort = hours < 15
            low_attendance = attendance < 70
            low_sleep = sleep_hours < 6
            active_tutoring = tutoring_sessions > 3

            # --- Rules for each predicted grade ---
            if predicted_grade in ["F"]:
                if low_effort:
                    advice_list.append("You need to study more hours per week to cover core topics.")
                if low_attendance:
                    advice_list.append("Prioritize attending lectures and tutorials.")
                if low_sleep:
                    advice_list.append("Improve sleep for better concentration.")
                advice_list.append("Focus on revision and assignments first to avoid failing.")

            elif predicted_grade in ["D", "D+"]:
                if low_effort:
                    advice_list.append("Increase study hours and focus on weak subjects.")
                if low_attendance:
                    advice_list.append("Attend classes regularly to catch up on missed content.")
                if active_tutoring:
                    advice_list.append("Make full use of tutoring sessions for targeted help.")
                advice_list.append("Focus on revision and assignments strategically.")

            elif predicted_grade in ["C-", "C", "C+"]:
                if high_effort:
                    advice_list.append("You are studying a lot; consider smarter study methods and revision strategies.")
                elif moderate_effort:
                    advice_list.append("Keep a steady study schedule and focus on difficult topics.")
                else:
                    advice_list.append("Increase study hours to improve your performance.")
                if low_sleep:
                    advice_list.append("Ensure you get enough rest to retain information.")
                advice_list.append("Prioritize assignments and revision according to difficulty.")

            elif predicted_grade in ["B-", "B", "B+"]:
                if high_effort:
                    advice_list.append("You are doing enough; focus on advanced topics or group work instead of more hours.")
                elif moderate_effort:
                    advice_list.append("Maintain consistent study hours and review challenging subjects.")
                else:
                    advice_list.append("Increase study hours slightly to reach a higher grade.")
                if active_tutoring:
                    advice_list.append("Use tutoring sessions for targeted improvement.")
                advice_list.append("Prioritize assignments, revision, and practice tests.")

            elif predicted_grade in ["A-", "A", "A+"]:
                if high_effort:
                    advice_list.append("Keep up your high effort; explore advanced topics, research, or group projects.")
                elif moderate_effort:
                    advice_list.append("Maintain current study habits and aim for high-quality learning.")
                else:
                    advice_list.append("Focus on maintaining quality study rather than increasing hours.")
                if low_sleep:
                    advice_list.append("Don’t sacrifice rest; good sleep is essential to stay sharp.")
                advice_list.append("Engage in extra learning or help peers to consolidate knowledge.")

            # --- ADVANCED CONTEXTUAL RULES (Secondary layer) ---
            # High effort but still low grade
            if predicted_grade in ["D", "D+", "C-", "C"] and high_effort:
                advice_list.append("You're putting in long hours, but results aren’t matching. Try changing your study techniques or focus areas instead of just adding more time.")

            # Low effort but decent grade
            if predicted_grade in ["B", "B+", "A-", "A"] and low_effort:
                advice_list.append("You’re performing well with less effort — great! But don’t get too relaxed; maintain consistency to keep it up.")

            # High effort + moderate grade
            if predicted_grade in ["B-", "B", "B+"] and high_effort:
                advice_list.append("You're working hard and it shows. Try smarter revision and active recall to push from ‘good’ to ‘excellent’.")

            # Low effort + low grade
            if predicted_grade in ["F", "D", "D+"] and low_effort:
                advice_list.append("Your current effort isn’t enough. Set up a clear study plan and stick to daily targets.")

            # High attendance + low grade
            if predicted_grade in ["D", "C-"] and attendance >= 80:
                advice_list.append("You’re attending classes but not grasping enough. Review your lecture notes and practice more.")

            # Low attendance + decent grades
            if predicted_grade in ["B", "A-", "A"] and low_attendance:
                advice_list.append("You manage good grades despite missing classes, but stay consistent — missing lectures may hurt future understanding.")

            # Low attendance + high effort
            if low_attendance and high_effort:
                advice_list.append("You're compensating for missed classes with extra study — good, but attending regularly would improve efficiency.")

            # High effort + low sleep
            if high_effort and low_sleep:
                advice_list.append("You’re studying a lot but losing rest. Quality study needs proper sleep — aim for at least 6–7 hours.")

            # Low effort + low sleep
            if low_effort and low_sleep:
                advice_list.append("You’re not studying enough *and* not sleeping well — both hurt learning. Start with proper rest to boost energy.")

            # Good sleep + high grades
            if predicted_grade in ["A-", "A", "A+"] and sleep_hours >= 7:
                advice_list.append("Your sleep balance is helping your high performance — keep that healthy rhythm.")

            # Attends tutoring but still low performance
            if active_tutoring and predicted_grade in ["D", "C-", "C"]:
                advice_list.append("You’re using tutoring, but results are low — maybe focus on asking specific questions or reviewing feedback more effectively.")

            # No tutoring but low grade
            if not active_tutoring and predicted_grade in ["F", "D", "C-"]:
                advice_list.append("Consider joining tutoring or group sessions — extra help can strengthen your weak areas.")

            # Tutoring + high effort + mid grade
            if active_tutoring and high_effort and predicted_grade in ["B-", "B"]:
                advice_list.append("Great use of tutoring and effort! Try shifting towards exam practice to fine-tune performance.")

            # Studying too much with lack of sleep
            if hours > 35 and low_sleep:
                advice_list.append("You’re at risk of burnout — take short breaks and manage your schedule better.")

            # Too much effort but still low grade
            if high_effort and predicted_grade in ["C-", "C", "C+"]:
                advice_list.append("You’re working hard but might be using inefficient methods. Try group discussions or active recall techniques.")

            # High grades but low sleep
            if predicted_grade in ["A-", "A", "A+"] and low_sleep:
                advice_list.append("Don’t sacrifice sleep — consistent rest is key to maintaining excellence.")

            # High grades but low attendance
            if predicted_grade in ["A-", "A", "A+"] and low_attendance:
                advice_list.append("Even high achievers need consistency — attend classes to stay updated and connected.")

            # High grades + high effort
            if predicted_grade in ["A", "A+"] and high_effort:
                advice_list.append("Outstanding effort! Explore leadership, mentoring, or research projects to expand your skills.")

            # Moderate grades + improving effort
            if predicted_grade in ["C+", "B-", "B"] and moderate_effort:
                advice_list.append("You’re on the right track — steady improvement will push your grade up further. Keep your momentum.")

            # Low grades but active tutoring
            if predicted_grade in ["D", "C-"] and active_tutoring:
                advice_list.append("You’re seeking help — that’s great! Stay consistent and track which tutoring sessions help you most.")

            # Balanced scenario (good performance, sleep, attendance)
            if predicted_grade in ["B+", "A-", "A"] and attendance >= 75 and 6 <= sleep_hours <= 8 and 15 <= hours <= 25:
                advice_list.append("Your habits are balanced — keep refining your learning style and aim for excellence.")

            # --- POST-PROCESSING CLEANUP ---
            final_advice = []
            for advice in advice_list:
                a_lower = advice.lower()
                duplicate = any(a_lower == existing.lower() for existing in final_advice)
                contradict = False

               # --- Contradiction filters ---
                for existing in final_advice:
                    e_lower = existing.lower()

                   # If student is already studying a lot, don’t tell them to study more
                    if ("study more" in a_lower or "increase study" in a_lower) and any(
                        phrase in e_lower for phrase in ["you are studying a lot", "you are doing enough", "smarter study"]
                    ):
                        contradict = True

                    # If already praised for good sleep, skip “improve sleep”
                    if ("sleep" in a_lower and "improve" in a_lower) and any(
                        phrase in e_lower for phrase in ["sleep balance", "sleep is helping", "good sleep"]
                    ):
                        contradict = True

                    # Avoid duplicate “attend classes” type advice
                    if ("attend" in a_lower or "lectures" in a_lower) and "attend" in e_lower and "consistency" in e_lower:
                        contradict = True

                # --- Add only if not duplicate/contradictory ---
                if not duplicate and not contradict:
                    final_advice.append(advice)

            # --- DISPLAY FINAL ADVICE ---
            st.success("💡 Suggested Actions Based on Your Situation:")
            for advice in final_advice:
                st.markdown(f"- {advice}")


            # ---------------- SENTIMENT ANALYSIS (VADER + CONTEXT) ----------------
            st.markdown("---")
            st.subheader("Tell me honestly — how do you feel about this grade?")

            with st.form(key="feedback_form"):
                fb = st.text_area(
                    "Your thoughts / feelings here...",
                    value=st.session_state.user_feedback,
                    placeholder="I feel motivated / a bit stressed / unsure...",
                    key="feedback_input",
                )
                submitted = st.form_submit_button("Share Feelings 💬")

                if submitted:
                    st.session_state.user_feedback = fb
                    if fb.strip() == "":
                        st.warning("⚠️ Please enter your thoughts before analyzing.")
                    else:
                        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                        analyzer = SentimentIntensityAnalyzer()

                        # Combine multiple data points for context
                        feedback_text = fb.lower()
                        grade = st.session_state.predicted_grade if "predicted_grade" in st.session_state else "N/A"
                        user_inputs = st.session_state.saved_inputs if "saved_inputs" in st.session_state else {}

                        # Prepare a context string combining grade + inputs
                        context_text = " ".join([f"{k}: {v}" for k, v in user_inputs.items()])
                        combined_text = f"{feedback_text} predicted grade: {grade} {context_text}"

                        # Perform sentiment analysis on combined text
                        score = analyzer.polarity_scores(combined_text)['compound']

                        # --- EMOTION CLASSIFICATION ---
                        if score >= 0.6:
                            emotion = "very positive"
                        elif 0.3 <= score < 0.6:
                            emotion = "positive"
                        elif 0.1 <= score < 0.3:
                            emotion = "neutral-positive"
                        elif -0.1 < score < 0.1:
                            emotion = "neutral"
                        elif -0.3 < score <= -0.1:
                            emotion = "negative"
                        elif -0.6 < score <= -0.3:
                            emotion = "very negative"
                        else:
                            emotion = "deeply negative"

                        # --- HUMAN-LIKE RESPONSES ---
                        responses = {
                            "very positive": [
                                "🔥 You sound *so* pumped — that’s exactly how winners think!",
                                "🌟 Your energy’s amazing! Keep believing in yourself, success follows that mindset.",
                                "😄 That’s pure motivation — your hard work will definitely pay off soon!"
                            ],
                            "positive": [
                                "💪 You’re feeling confident, huh? Keep that vibe going strong!",
                                "🙌 Great to see positivity! Stay on track, and results will follow.",
                                "✨ Loving that optimistic tone — you’re in control of your growth!"
                            ],
                            "neutral-positive": [
                                "🙂 You seem calm and hopeful — that’s a great mindset to move forward.",
                                "🧘 Balanced thinking is powerful. Keep reflecting like this, it helps long-term.",
                                "😌 You’re doing okay, keep up your steady rhythm!"
                            ],
                            "neutral": [
                                "😶 Sounds neutral — maybe you’re still processing the advice?",
                                "🤔 That’s fine. Take time to think things through — clarity always comes.",
                                "🕊️ Even small reflections like this can lead to big realizations."
                            ],
                            "negative": [
                                "😔 You seem worried. That’s okay — setbacks are temporary, not final.",
                                "💙 Don’t let low grades define you — use them as a push forward.",
                                "🌧️ It’s okay to struggle; even the best do. You’ve got this, step by step."
                            ],
                            "very negative": [
                                "💔 Seems like you’re really down. Take a break — your mind deserves rest too.",
                                "😞 Grades don’t show your full potential. You’re still growing, trust that.",
                                "🫶 You’re being hard on yourself. Remember — progress isn’t linear, but it’s real."
                            ],
                            "deeply negative": [
                                "🥺 You sound truly exhausted. Please take care of yourself — the rest can wait.",
                                "😢 It’s okay. You’ve already come far — don’t stop now.",
                                "🫂 You’re not alone in this. Rest, reset, and rise again tomorrow."
                            ]
                        }

                        # --- CONTEXT-AWARE ADJUSTMENTS ---
                        extra = ""
                        if isinstance(grade, (int, float)):
                            if grade < 50:
                                extra += " 📉 I know that grade looks tough, but it’s not the end — it’s just the start of improvement."
                            elif grade < 70:
                                extra += " 💪 You’re doing well — with a bit more focus, you can ace it next time."
                            else:
                                extra += " 🏆 Excellent result! Keep that consistency — you’re clearly mastering this."

                        # emotional keyword check
                        neg_words = ["tired", "worried", "stressed", "anxious", "exhausted", "sad", "depressed", "low", "hopeless"]
                        pos_words = ["motivated", "confident", "excited", "happy", "grateful", "ready", "energized"]

                        if any(word in feedback_text for word in neg_words):
                            extra += " 💖 Don’t forget to take care of yourself — mental rest boosts performance too."
                        elif any(word in feedback_text for word in pos_words):
                            extra += " 🌈 Love that energy — keep feeding that momentum!"

                        # --- FINAL MESSAGE ---
                        msg = random.choice(responses[emotion]) + extra
                        st.session_state.analysis_msg = msg

            # Show the last message
            if st.session_state.analysis_msg:
                st.info(f"💬 **{st.session_state.analysis_msg}**")

        st.markdown('</div>', unsafe_allow_html=True)






# ---------------- UPDATE FORM PAGE ----------------
def show_update_grade_page():
    st.title("📘 Update Your Actual Grade")

    # --- Path to data folder ---
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    data_file = os.path.join(data_folder, "predictions.csv")

    if not os.path.exists(data_file):
        st.warning("No predictions found yet.")
        return

    df = pd.read_csv(data_file)

    user_data = df[df["Username"] == st.session_state.username]
    if user_data.empty:
        st.info("You haven’t made any predictions yet.")
        if st.button("Back to Predictions"):
            st.session_state.page = "form"
            st.rerun()
        return

    display_data = user_data[["Timestamp", "subject_name", "Predicted_Grade", "Actual_Grade"]].reset_index(drop=True)
    display_data.index += 1  # Make index start from 1
    st.dataframe(display_data)

    # Create a display label combining time and subject
    user_data["Display_Label"] = user_data["Timestamp"] + " — " + user_data["subject_name"]

    selected_label = st.selectbox(
        "Select a prediction to update",
        user_data["Display_Label"]
    )

    # Extract the actual timestamp value back from the selected label
    selected_time = user_data.loc[user_data["Display_Label"] == selected_label, "Timestamp"].values[0]

    actual_grade = st.selectbox("Enter your actual grade", ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "F"])

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Save Actual Grade"):
            df.loc[(df["Username"] == st.session_state.username) & (df["Timestamp"] == selected_time), "Actual_Grade"] = actual_grade
            df.to_csv(data_file, index=False)
            st.rerun()

    with col3:
        if st.button("Back to Predictions"):
            st.session_state.page = "form"
            st.rerun()

    # ---------- TREND CHART ----------
    st.markdown("<h5 style='text-align:center; color:white;'>Your Prediction Trend Over Time</h5>", unsafe_allow_html=True)

    # Filter out rows with actual grade entered
    graded_data = user_data[user_data["Actual_Grade"].notna() & (user_data["Actual_Grade"] != "")]

    if len(graded_data) >= 1:
        # Combine Timestamp and Subject Name for x-axis
        graded_data["X_Label"] = graded_data["Timestamp"] + "<br>" + graded_data["subject_name"]

        # Convert grades to numeric scale for visualization
        grade_scale = {"A+": 12, "A": 11, "A-": 10, "B+": 9, "B": 8, "B-": 7,
                   "C+": 6, "C": 5, "C-": 4, "D+": 3, "D": 2, "F": 1}
        graded_data["Predicted_Score"] = graded_data["Predicted_Grade"].map(grade_scale)
        graded_data["Actual_Score"] = graded_data["Actual_Grade"].map(grade_scale)

        fig = go.Figure()

        # Add predicted line
        fig.add_trace(go.Scatter(
            x=graded_data["X_Label"],
            y=graded_data["Predicted_Score"],
            mode='lines+markers',
            name='Predicted Grade',
            line=dict(color='deepskyblue', width=2)
        ))

        # Add actual line
        fig.add_trace(go.Scatter(
            x=graded_data["X_Label"],
            y=graded_data["Actual_Score"],
            mode='lines+markers',
            name='Actual Grade',
            line=dict(color='lightgreen', width=2, dash='dot')
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Prediction Time",
            yaxis_title="Grade Level",
            yaxis=dict(range=[0, 13], tickvals=list(grade_scale.values()), 
                       ticktext=list(grade_scale.keys()), color='white'),
            xaxis=dict(color='white'),
            legend=dict(font=dict(color='white')),
            margin=dict(t=30, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart compares your predicted and actual grades over time, helping you track prediction accuracy.")
    else:
        st.info("No actual grades entered yet — update one to view your trend chart.")
    # ---------- END TREND CHART ----------







# ---------------- ROUTING ----------------
if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "login":
    show_login_page()
elif st.session_state.page == "signup":
    show_signup_page()
elif st.session_state.page == "form":
    if st.session_state.username:
        show_form_page()
    else:
        st.session_state.page = "login"
        st.rerun()
elif st.session_state.page == "update_grade":
    show_update_grade_page()
elif st.session_state.page == "model_details":
    st.empty()
    show_model_details_page()
elif st.session_state.page == "contact_us":
    show_contact_page()

