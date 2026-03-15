# Smart Grid Load Predicton

The Smart Grid Load Predictor is a machine learning and data visualization project developed to predict electricity grid load using Electric Vehicle (EV) charging station data. 
As EV usage increases, charging demand can significantly impact electricity grids. This project analyzes charging activity, renewable energy usage, and time-based patterns to estimate future grid load.

The application is built using Python and Streamlit and provides an interactive dashboard where users can explore data, visualize electricity demand trends, and generate predictions.
A Random Forest Regression model is used to learn patterns from historical data and forecast grid load.

## Features

-Interactive Streamlit dashboard for data exploration

-Machine learning model to predict grid load

-Visualization of EV charging demand patterns

-Actual vs Predicted grid load comparison

-Feature importance analysis

-Anomaly detection for unusual grid load events

-Geospatial visualization of grid load by city zone

-Seasonal decomposition to analyze demand trends

## Machine Learning Model

-Algorithm Used: Random Forest Regressor

-Target Variable: Grid Load (MW)

### Model Evaluation Metrics

-MAE (Mean Absolute Error)

-R² Score

-MAPE (Mean Absolute Percentage Error)

#### Project Structure
```
smart-grid-load-predictor
│
├── app.py
├── ev_charging_station_usage_grid_load.csv
├── scaler.pkl
├── requirements.txt
└── README.md
```

##### How to Run the Project

1️. Clone the Repository
```
git clone https://github.com/AyshaMehrin572/smart-grid-load-predictor.git
cd smart-grid-load-predictor
```

2️. Install Required Libraries
```
pip install -r requirements.txt
```
3️. Run the Streamlit Application
```
streamlit run app.py
```
After running the command, the application will open in your browser:
```
http://localhost:8501
```
###### Output

-The application generates an interactive dashboard that provides:

-Grid load prediction based on EV charging data

-Visual analysis of electricity demand patterns

-Comparison between actual and predicted grid load

-Identification of unusual grid load behavior

-Geographic visualization of electricity load distribution

-Seasonal patterns and trends in electricity consumption
