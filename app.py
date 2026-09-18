import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AWS Anomaly Detection",
    page_icon="🌦️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🌦️ AWS Intelligent Anomaly Detection System")

st.write(
    "AI/ML-based system for detecting abnormal "
    "Automatic Weather Station readings."
)


# --------------------------------------------------
# STEP 1: LOAD WEATHER DATA
# --------------------------------------------------

data = pd.read_csv("aws_weather_data.csv")


# --------------------------------------------------
# STEP 2: CONVERT TIMESTAMP
# --------------------------------------------------

data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)


# --------------------------------------------------
# STEP 3: SELECT WEATHER PARAMETERS
# --------------------------------------------------

features = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "rainfall"
]


# --------------------------------------------------
# STEP 4: CHECK FOR MISSING VALUES
# --------------------------------------------------

missing_before = data[features].isnull().sum().sum()


# Replace missing values with median
data[features] = data[features].fillna(
    data[features].median()
)


# --------------------------------------------------
# STEP 8: PREPARE FEATURES FOR MACHINE LEARNING
# --------------------------------------------------

X = data[features].copy()


# --------------------------------------------------
# STEP 9: AI/ML ANOMALY DETECTION
# --------------------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

model.fit(X)

predictions = model.predict(X)


# --------------------------------------------------
# STEP 10: NORMAL / ANOMALY RESULT
# --------------------------------------------------

data["prediction"] = predictions

data["status"] = data["prediction"].map({
    1: "Normal",
    -1: "Anomaly"
})


# --------------------------------------------------
# STEP 11: ANOMALY SCORE + REASON
# --------------------------------------------------

data["anomaly_score"] = model.decision_function(X)


def find_reason(row):

    reasons = []

    if row["temperature"] > 45 or row["temperature"] < 10:
        reasons.append("Temperature unusual")

    if row["humidity"] > 95 or row["humidity"] < 20:
        reasons.append("Humidity unusual")

    if row["pressure"] > 1050 or row["pressure"] < 970:
        reasons.append("Pressure unusual")

    if row["wind_speed"] > 60:
        reasons.append("Wind speed unusual")

    if row["rainfall"] > 50:
        reasons.append("Rainfall unusual")

    if len(reasons) == 0:
        return "No unusual parameter detected"

    return ", ".join(reasons)


data["reason"] = data.apply(
    find_reason,
    axis=1
)


# --------------------------------------------------
# STEP 14: ALERT + SEVERITY
# --------------------------------------------------

def get_severity(row):

    if row["status"] == "Normal":
        return "Normal"

    unusual_count = 0

    if row["temperature"] > 45 or row["temperature"] < 10:
        unusual_count += 1

    if row["humidity"] > 95 or row["humidity"] < 20:
        unusual_count += 1

    if row["pressure"] > 1050 or row["pressure"] < 970:
        unusual_count += 1

    if row["wind_speed"] > 60:
        unusual_count += 1

    if row["rainfall"] > 50:
        unusual_count += 1

    if unusual_count >= 3:
        return "Critical"

    if unusual_count >= 2:
        return "High"

    return "Medium"


data["severity"] = data.apply(
    get_severity,
    axis=1
)


# --------------------------------------------------
# ALERT SUMMARY
# --------------------------------------------------

critical_count = (
    data["severity"] == "Critical"
).sum()

high_count = (
    data["severity"] == "High"
).sum()

medium_count = (
    data["severity"] == "Medium"
).sum()


st.subheader("🚨 Alert Summary")


alert1, alert2, alert3 = st.columns(3)


with alert1:
    st.metric(
        "Critical Alerts",
        critical_count
    )


with alert2:
    st.metric(
        "High Alerts",
        high_count
    )


with alert3:
    st.metric(
        "Medium Alerts",
        medium_count
    )


if critical_count > 0:

    st.error(
        f"🚨 {critical_count} critical weather anomaly(s) detected!"
    )

elif high_count > 0:

    st.warning(
        f"⚠️ {high_count} high-severity anomaly(s) detected."
    )

else:

    st.success(
        "✅ No critical or high-severity anomalies detected."
    )


# --------------------------------------------------
# STEP 12: DASHBOARD SUMMARY
# --------------------------------------------------

total_records = len(data)

anomaly_count = (
    data["status"] == "Anomaly"
).sum()

normal_count = (
    data["status"] == "Normal"
).sum()


st.subheader("📌 System Summary")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Records",
        total_records
    )


with col2:
    st.metric(
        "Normal Readings",
        normal_count
    )


with col3:
    st.metric(
        "Anomalies Detected",
        anomaly_count
    )


# --------------------------------------------------
# TEMPERATURE MONITORING
# --------------------------------------------------

st.subheader("📈 Temperature Monitoring")


fig = px.line(
    data,
    x="timestamp",
    y="temperature",
    title="Temperature Over Time"
)


st.plotly_chart(
    fig,
    use_container_width=True,
    key="temperature_chart"
)


# --------------------------------------------------
# STEP 13: WEATHER GRAPHS + ANOMALY VIEW
# --------------------------------------------------

st.subheader("🌦️ Weather Parameter Monitoring")


parameter = st.selectbox(
    "Select a weather parameter",
    [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "rainfall"
    ]
)


fig2 = px.line(
    data,
    x="timestamp",
    y=parameter,
    title=f"{parameter.replace('_', ' ').title()} Over Time"
)


st.plotly_chart(
    fig2,
    use_container_width=True,
    key="weather_parameter_chart"
)


# --------------------------------------------------
# DETECTED ANOMALIES ONLY
# --------------------------------------------------

st.subheader("🚨 Detected Anomalies Only")


anomalies = data[
    data["status"] == "Anomaly"
]


st.dataframe(
    anomalies[
        [
            "timestamp",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "rainfall",
            "status",
            "anomaly_score",
            "reason",
            "severity"
        ]
    ],
    use_container_width=True
)


# --------------------------------------------------
# ANOMALY DETECTION RESULTS
# --------------------------------------------------

st.subheader("🚨 Anomaly Detection Results")


st.dataframe(
    data[
        [
            "timestamp",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "rainfall",
            "status",
            "anomaly_score",
            "reason",
            "severity"
        ]
    ],
    use_container_width=True
)


# --------------------------------------------------
# STEP 8: ML INPUT FEATURES DISPLAY
# --------------------------------------------------

st.subheader("🧠 ML Input Features")


st.write(
    "The machine-learning model uses the following "
    "weather parameters to identify unusual patterns:"
)


st.write(
    "Temperature, Humidity, Pressure, Wind Speed and Rainfall"
)


st.dataframe(
    X.head(10),
    use_container_width=True
)


# --------------------------------------------------
# MISSING VALUES AFTER CLEANING
# --------------------------------------------------

missing_after = data[features].isnull().sum().sum()


# --------------------------------------------------
# SHOW DATA INFORMATION
# --------------------------------------------------

st.subheader("📊 Weather Dataset")


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Records",
        len(data)
    )


with col2:
    st.metric(
        "Missing Values Before Cleaning",
        missing_before
    )


with col3:
    st.metric(
        "Missing Values After Cleaning",
        missing_after
    )


# Display cleaned data
st.dataframe(
    data,
    use_container_width=True
)