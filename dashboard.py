import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------- TITLE --------------------
st.title("⚡ EV Charging Station & Grid Load Analysis")

# -------------------- LOAD DATA --------------------
df = pd.read_csv("ev_charging_station_usage_grid_load.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.write("Shape:", df.shape)
st.write("Missing Values:", df.isnull().sum())

df = df.drop_duplicates()

# -------------------- DATE FEATURES --------------------
df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday

# -------------------- ENCODE CATEGORICAL --------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# -------------------- FEATURE ENGINEERING --------------------
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 21) else 0)

df['energy_per_vehicle'] = df['energy_dispensed_kwh'] / df['vehicles_charged'].replace(0,1)
df['duration_per_vehicle'] = df['avg_charging_duration_minutes'] / df['vehicles_charged'].replace(0,1)
df['load_per_vehicle'] = df['grid_load_mw'] / df['vehicles_charged'].replace(0,1)

df['renewable_usage_score'] = df['renewable_energy_used_percent'] * df['energy_dispensed_kwh']

# -------------------- KPI METRICS --------------------
st.subheader("Key Metrics")
col1, col2 = st.columns(2)

col1.metric("Total Energy Used (kWh)", round(df['energy_dispensed_kwh'].sum(),2))
col2.metric("Total Vehicles Charged", int(df['vehicles_charged'].sum()))

# -------------------- HISTOGRAMS --------------------
st.subheader("Distribution of Features")
fig, ax = plt.subplots(figsize=(12,8))
df.hist(ax=ax)
st.pyplot(fig)

# -------------------- BOXPLOTS --------------------
st.subheader("Grid Load vs Risk")
fig, ax = plt.subplots()
sns.boxplot(x='peak_load_risk', y='grid_load_mw', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Vehicles Charged vs Risk")
fig, ax = plt.subplots()
sns.boxplot(x='peak_load_risk', y='vehicles_charged', data=df, ax=ax)
st.pyplot(fig)

st.subheader("Energy Dispensed vs Risk")
fig, ax = plt.subplots()
sns.boxplot(x='peak_load_risk', y='energy_dispensed_kwh', data=df, ax=ax)
st.pyplot(fig)

# -------------------- COUNT PLOTS --------------------
st.subheader("Station Type vs Risk")
fig, ax = plt.subplots()
sns.countplot(x='station_type', hue='peak_load_risk', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("City Zone vs Risk")
fig, ax = plt.subplots()
sns.countplot(x='city_zone', hue='peak_load_risk', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------- CORRELATION HEATMAP --------------------
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# -------------------- MACHINE LEARNING --------------------
X = df.drop('peak_load_risk', axis=1)
y = df['peak_load_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X.drop('date_time', axis=1), y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC()
}

st.subheader("Model Accuracy Comparison")

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    st.write(f"{name} Accuracy:", round(acc,3))

# -------------------- ACCURACY BAR CHART --------------------
fig, ax = plt.subplots()
ax.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
st.pyplot(fig)

# -------------------- BEST MODEL --------------------
best_model = max(results, key=results.get)
st.success(f"Best Performing Model: {best_model}")

# -------------------- CONFUSION MATRIX --------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
final_pred = rf.predict(X_test)

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, final_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, final_pred))