import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import datetime
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------
# Page configuration
st.set_page_config(page_title="Smart Grid Load Predictor", layout="wide")
st.title("⚡ Smart Grid Load Prediction System")
st.markdown("Predict future grid load based on EV charging demand, time patterns, and renewable energy usage.")

# ------------------------------
# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("ev_charging_station_usage_grid_load.csv", parse_dates=["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    return df

# Load scaler (cached)
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

df = load_data()
scaler = load_scaler()

# ------------------------------
# Feature engineering function
def engineer_features(df):
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['energy_per_min'] = df['energy_dispensed_kwh'] / df['avg_charging_duration_minutes'].replace(0, np.nan)
    df = df.sort_values('date_time').reset_index(drop=True)
    df['rolling_3hr_load'] = df['grid_load_mw'].rolling(window=3, min_periods=1).mean()
    df['rolling_6hr_load'] = df['grid_load_mw'].rolling(window=6, min_periods=1).mean()
    df['lag_1'] = df['grid_load_mw'].shift(1)
    df['lag_2'] = df['grid_load_mw'].shift(2)
    df['lag_24'] = df['grid_load_mw'].shift(24)
    df['hour_energy_interaction'] = df['hour'] * df['energy_dispensed_kwh']
    df = df.dropna().reset_index(drop=True)
    return df

df_feat = engineer_features(df)

feature_names = ['record_id', 'vehicles_charged', 'avg_charging_duration_minutes',
                 'energy_dispensed_kwh', 'grid_load_mw', 'renewable_energy_used_percent',
                 'is_weekend', 'is_peak_hour', 'energy_per_min', 'rolling_3hr_load',
                 'rolling_6hr_load', 'lag_1', 'lag_2', 'lag_24', 'hour_energy_interaction']
selected_features = [f for f in feature_names if f not in ['record_id', 'grid_load_mw']]
X = df_feat[selected_features]
y = df_feat['grid_load_mw']

# Train/test split
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df_feat.index, test_size=0.2, random_state=42
)

# ------------------------------
# Train model (cached)
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Predictions on test set (for analytics)
y_pred_test = model.predict(X_test)
residuals = y_test.values - y_pred_test
std_resid = residuals.std()

# ------------------------------
# Sidebar – Navigation
st.sidebar.header("Navigation")
pages = [
    "Data Explorer",
    "Predict Grid Load",
    "Actual vs Predicted",
    "Feature Importance",
    "Anomaly Detection",
    "Geospatial Map",
    "Seasonal Decomposition"
]
page = st.sidebar.radio("Go to", pages)

# ------------------------------
# Helper to get test data with datetime
test_df = df_feat.loc[idx_test].copy()
test_df['predicted_load'] = y_pred_test
test_df['residual'] = residuals

# ------------------------------
# 1. Data Explorer (original)
if page == "Data Explorer":
    st.header("📊 Data Exploration")
    
    with st.expander("Show raw data sample"):
        st.dataframe(df.head(100))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Grid Load (MW)", f"{df['grid_load_mw'].mean():.1f}")
    col3.metric("Max Grid Load (MW)", f"{df['grid_load_mw'].max():.1f}")
    col4.metric("Avg Renewable %", f"{df['renewable_energy_used_percent'].mean():.1f}%")
    
    st.subheader("Grid Load Over Time by City Zone")

    # --- Controls ---
    all_zones = sorted(df['city_zone'].dropna().unique().tolist())
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        chart_granularity = st.selectbox(
            "Aggregation", ["Daily Average", "Hourly Average"], index=0,
            help="Daily is cleaner; hourly shows intra-day patterns."
        )
    with col_f2:
        selected_zones = st.multiselect(
            "City zones to display", all_zones, default=all_zones,
            help="Deselect zones to reduce clutter."
        )

    if not selected_zones:
        st.warning("Please select at least one city zone.")
    else:
        freq = 'D' if chart_granularity == "Daily Average" else 'h'
        zone_df = df[df['city_zone'].isin(selected_zones)].copy()
        agg_df = (
            zone_df
            .groupby([pd.Grouper(key='date_time', freq=freq), 'city_zone'])['grid_load_mw']
            .mean()
            .reset_index()
        )
        agg_df.columns = ['date_time', 'City Zone', 'Grid Load (MW)']

        # View 1 – Time-series line chart (daily/hourly)
        fig_line = px.line(
            agg_df, x='date_time', y='Grid Load (MW)', color='City Zone',
            title=f"{chart_granularity} Grid Load by City Zone",
            markers=(freq == 'D'),
        )
        fig_line.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Avg Grid Load (MW)",
            legend_title="City Zone",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # View 2 – Hourly heatmap: avg load by Hour-of-day × City Zone
        st.markdown("#### 🕐 When Does Each Zone Peak? (Hour-of-Day Pattern)")
        heat_df = (
            zone_df
            .assign(hour=zone_df['date_time'].dt.hour)
            .groupby(['city_zone', 'hour'])['grid_load_mw']
            .mean()
            .reset_index()
        )
        heat_pivot = heat_df.pivot(index='city_zone', columns='hour', values='grid_load_mw')
        fig_heat = px.imshow(
            heat_pivot,
            labels=dict(x="Hour of Day", y="City Zone", color="Avg Load (MW)"),
            color_continuous_scale="YlOrRd",
            title="Average Grid Load (MW) — Hour of Day × City Zone",
            aspect="auto",
            text_auto=".0f",
        )
        fig_heat.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_heat, use_container_width=True)

        # View 3 – Simple bar chart summary
        st.markdown("#### 📊 Overall Average Load per Zone")
        bar_df = (
            zone_df.groupby('city_zone')['grid_load_mw']
            .mean()
            .reset_index()
            .sort_values('grid_load_mw', ascending=False)
        )
        bar_df.columns = ['City Zone', 'Avg Grid Load (MW)']
        fig_bar = px.bar(
            bar_df, x='City Zone', y='Avg Grid Load (MW)',
            color='City Zone', text_auto='.1f',
            title="Average Grid Load per City Zone (Full Dataset)",
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, yaxis_title="Avg Grid Load (MW)")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("Grid Load Distribution by Station Type")
    fig = px.box(df, x='station_type', y='grid_load_mw', color='station_type',
                 title="Grid Load (MW) per Station Type")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Peak Load Risk Frequency")
    risk_counts = df['peak_load_risk'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']
    fig = px.pie(risk_counts, values='Count', names='Risk', title="Proportion of Peak Load Risk")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 2. Predict Grid Load (original, updated with model metrics)
elif page == "Predict Grid Load":
    st.header("🔮 Predict Future Grid Load")
    st.markdown("Adjust the parameters below to simulate an hour and see the predicted grid load.")
    
    # Show model performance
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    st.info(f"**Model Performance on Test Set**  |  MAE: {mae:.2f} MW  |  R²: {r2:.3f}")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hour = st.slider("Hour of day", 0, 23, 12)
            vehicles = st.number_input("Vehicles charged", min_value=0, max_value=50, value=10)
            duration = st.number_input("Avg charging duration (min)", min_value=0.0, max_value=200.0, value=60.0)
            energy = st.number_input("Energy dispensed (kWh)", min_value=0.0, max_value=500.0, value=200.0)
        
        with col2:
            renewable = st.slider("Renewable energy used (%)", 0.0, 100.0, 45.0)
            date_input = st.date_input("Date", datetime.date.today())
            is_weekend = 1 if date_input.weekday() >= 5 else 0
            st.write(f"Weekend? {'Yes' if is_weekend else 'No'}")
            is_peak = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
            st.write(f"Peak hour? {'Yes' if is_peak else 'No'}")
        
        with col3:
            st.markdown("**Recent grid loads (required for lags & rolling)**")
            lag_1 = st.number_input("Load 1 hour ago (MW)", value=300.0)
            lag_2 = st.number_input("Load 2 hours ago (MW)", value=295.0)
            lag_3 = st.number_input("Load 3 hours ago (MW)", value=290.0)
            lag_24 = st.number_input("Load 24 hours ago (MW)", value=310.0)
            rolling_3hr = (lag_1 + lag_2 + lag_3) / 3
            # For rolling 6hr, we need 6 values; for demo, approximate by repeating last three
            rolling_6hr = (lag_1 + lag_2 + lag_3 + lag_1 + lag_2 + lag_3) / 6
            st.info("Rolling 6hr approximated with last three loads repeated.")
        
        submitted = st.form_submit_button("Predict Grid Load")
    
    if submitted:
        energy_per_min = energy / duration if duration > 0 else 0
        hour_energy_interaction = hour * energy
        
        features = np.array([[
            vehicles, duration, energy, renewable,
            is_weekend, is_peak, energy_per_min,
            rolling_3hr, rolling_6hr, lag_1, lag_2, lag_24,
            hour_energy_interaction
        ]])
        
        pred_load = model.predict(features)[0]
        
        st.success(f"### Predicted Grid Load: **{pred_load:.2f} MW**")
        st.json({
            "Hour": hour,
            "Vehicles": vehicles,
            "Duration (min)": duration,
            "Energy (kWh)": energy,
            "Renewable %": renewable,
            "Weekend": bool(is_weekend),
            "Peak hour": bool(is_peak),
            "Lag 1h": lag_1,
            "Lag 2h": lag_2,
            "Lag 3h": lag_3,
            "Lag 24h": lag_24
        })

# ------------------------------
# 3. Actual vs Predicted Time Series
elif page == "Actual vs Predicted":
    st.header("📈 Actual vs Predicted Grid Load (Test Set)")

    # Sort test_df by time so line charts make sense
    td = test_df.sort_values('date_time').copy()

    # ── Metrics row ────────────────────────────────────────────────────────
    mae_all  = mean_absolute_error(td['grid_load_mw'], td['predicted_load'])
    r2_all   = r2_score(td['grid_load_mw'], td['predicted_load'])
    mape_all = (np.abs((td['grid_load_mw'] - td['predicted_load']) / td['grid_load_mw'].replace(0, np.nan)).mean()) * 100
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE (full test set)", f"{mae_all:.2f} MW")
    mc2.metric("R² Score", f"{r2_all:.4f}", help="1.0 = perfect prediction")
    mc3.metric("MAPE", f"{mape_all:.1f}%", help="Mean Absolute Percentage Error")

    st.divider()

    # ── View 1 – Daily-average time-series with error band ─────────────────
    st.subheader("① Daily Average: Actual vs Predicted")
    st.caption("Aggregating to daily averages removes point-to-point noise and reveals the overall trend clearly.")

    min_date = td['date_time'].min().date()
    max_date = td['date_time'].max().date()
    default_start = max_date - datetime.timedelta(days=14)
    if default_start < min_date:
        default_start = min_date
    date_range = st.slider("Date range", min_date, max_date, (default_start, max_date))

    mask = (td['date_time'].dt.date >= date_range[0]) & (td['date_time'].dt.date <= date_range[1])
    filt = td[mask].copy()

    daily = (
        filt.set_index('date_time')
        .resample('D')[['grid_load_mw', 'predicted_load', 'residual']]
        .agg({'grid_load_mw': 'mean', 'predicted_load': 'mean', 'residual': 'std'})
        .reset_index()
        .dropna()
    )
    daily.columns = ['date', 'actual', 'predicted', 'resid_std']
    daily['upper'] = daily['predicted'] + daily['resid_std']
    daily['lower'] = daily['predicted'] - daily['resid_std']

    fig_ts = go.Figure()
    # Confidence band
    fig_ts.add_trace(go.Scatter(
        x=pd.concat([daily['date'], daily['date'][::-1]]),
        y=pd.concat([daily['upper'], daily['lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,100,100,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip', name='Prediction ±1σ'
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily['date'], y=daily['actual'],
        mode='lines+markers', name='Actual',
        line=dict(color='#2196F3', width=2), marker=dict(size=5)
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily['date'], y=daily['predicted'],
        mode='lines+markers', name='Predicted',
        line=dict(color='#F44336', width=2, dash='dash'), marker=dict(size=5)
    ))
    fig_ts.update_layout(
        hovermode='x unified',
        xaxis_title='Date', yaxis_title='Avg Grid Load (MW)',
        legend=dict(orientation='h', y=1.1),
        height=380,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Period-specific metrics
    if len(filt) >= 2:
        mae_p = mean_absolute_error(filt['grid_load_mw'], filt['predicted_load'])
        r2_p  = r2_score(filt['grid_load_mw'], filt['predicted_load'])
        pc1, pc2 = st.columns(2)
        pc1.metric("MAE (selected period)", f"{mae_p:.2f} MW", delta=f"{mae_p - mae_all:+.2f} vs overall")
        pc2.metric("R² (selected period)",  f"{r2_p:.4f}")

    st.divider()

    # ── View 2 – Scatter: Actual vs Predicted ──────────────────────────────
    st.subheader("② Scatter Plot — How Close Is the Model?")
    st.caption("Points on or near the dashed line mean the model is accurate. Spread = error.")

    fig_sc = px.scatter(
        filt, x='grid_load_mw', y='predicted_load',
        color='residual', color_continuous_scale='RdYlGn_r',
        labels={'grid_load_mw': 'Actual Load (MW)', 'predicted_load': 'Predicted Load (MW)',
                'residual': 'Residual (MW)'},
        opacity=0.6, height=380,
        title="Actual vs Predicted (each point = one test record)"
    )
    # Perfect-fit reference line
    lo = min(filt['grid_load_mw'].min(), filt['predicted_load'].min())
    hi = max(filt['grid_load_mw'].max(), filt['predicted_load'].max())
    fig_sc.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi,
                     line=dict(color='black', dash='dash', width=1.5))
    fig_sc.add_annotation(x=hi, y=hi, text="Perfect fit", showarrow=False,
                           yshift=10, font=dict(size=11, color='black'))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # ── View 3 – Residual distribution ─────────────────────────────────────
    st.subheader("③ Residual Distribution — Where Does the Model Err?")
    st.caption("A bell curve centred at 0 means errors are balanced. A shifted curve means systematic over/under-prediction.")

    fig_res = px.histogram(
        filt, x='residual', nbins=40,
        color_discrete_sequence=['#7C4DFF'],
        labels={'residual': 'Residual (Actual − Predicted, MW)'},
        title="Distribution of Prediction Errors"
    )
    fig_res.add_vline(x=0, line_dash='dash', line_color='black', annotation_text='Zero error', annotation_position='top right')
    mean_resid = filt['residual'].mean()
    fig_res.add_vline(x=mean_resid, line_color='red',
                      annotation_text=f'Mean error: {mean_resid:.1f} MW',
                      annotation_position='top left')
    fig_res.update_layout(height=320)
    st.plotly_chart(fig_res, use_container_width=True)


# ------------------------------
# 4. Feature Importance
elif page == "Feature Importance":
    st.header("🌟 Feature Importance (Random Forest)")
    
    importances = model.feature_importances_
    feat_names = X.columns
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
    
    fig = px.bar(imp_df, x='importance', y='feature', orientation='h',
                 title="Feature Importances",
                 labels={'importance': 'Importance', 'feature': ''},
                 color='importance', color_continuous_scale='viridis')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Interpretation:** Higher values indicate more influence on grid load prediction.")

# ------------------------------
# 5. Anomaly Detection
elif page == "Anomaly Detection":
    st.header("⚠️ Anomaly Detection")
    
    threshold = st.slider("Residual threshold (standard deviations)", 1.0, 3.0, 2.0, 0.1)
    anomalies = test_df[np.abs(test_df['residual']) > threshold * std_resid]
    
    st.write(f"Found **{len(anomalies)}** anomalies (residual > {threshold:.1f}σ).")
    
    if not anomalies.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df['date_time'], y=test_df['grid_load_mw'],
                                  mode='markers', name='Normal', marker=dict(color='lightgray', size=4)))
        fig.add_trace(go.Scatter(x=anomalies['date_time'], y=anomalies['grid_load_mw'],
                                  mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        fig.update_layout(title="Anomalies Highlighted", xaxis_title="Time", yaxis_title="Load (MW)")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Show anomaly details"):
            st.dataframe(anomalies[['date_time', 'grid_load_mw', 'predicted_load', 'residual'] + 
                                     ['vehicles_charged', 'renewable_energy_used_percent', 'is_peak_hour']])
    else:
        st.info("No anomalies found with the current threshold.")

# ------------------------------
# 6. Geospatial Map (with dummy coordinates)
elif page == "Geospatial Map":
    st.header("🗺️ Geospatial View (Simulated)")
    
    # Dummy coordinates for each zone (approximate city zones)
    zone_coords = {
        'Central': {'lat': 40.7128, 'lon': -74.0060},
        'North':   {'lat': 40.8000, 'lon': -74.0200},
        'South':   {'lat': 40.6000, 'lon': -74.0100},
        'East':    {'lat': 40.7150, 'lon': -73.9800},
        'West':    {'lat': 40.7100, 'lon': -74.0500}
    }
    
    # Aggregate latest data per zone (last hour in dataset)
    last_time = df['date_time'].max()
    latest = df[df['date_time'] == last_time].groupby('city_zone').agg({
        'grid_load_mw': 'mean',
        'peak_load_risk': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()
    
    # Add coordinates
    latest['lat'] = latest['city_zone'].map(lambda z: zone_coords[z]['lat'])
    latest['lon'] = latest['city_zone'].map(lambda z: zone_coords[z]['lon'])
    
    # Risk color mapping
    risk_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    latest['color'] = latest['peak_load_risk'].map(risk_color)
    
    fig = px.scatter_mapbox(latest, lat='lat', lon='lon', size='grid_load_mw',
                            color='peak_load_risk', color_discrete_map=risk_color,
                            hover_name='city_zone', hover_data={'grid_load_mw': True},
                            zoom=10, mapbox_style='carto-positron',
                            title=f"Latest Grid Load by Zone ({last_time})")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("Note: Coordinates are approximate for demonstration. Size of bubble represents grid load.")

# ------------------------------
# 7. Seasonal Decomposition
elif page == "Seasonal Decomposition":
    st.header("📉 Seasonal Decomposition of Grid Load")
    
    # Resample to daily average for cleaner decomposition (or keep hourly, but period=24)
    daily_load = df.set_index('date_time')['grid_load_mw'].resample('D').mean().dropna()
    
    # Perform decomposition (additive model)
    decomposition = seasonal_decompose(daily_load, model='additive', period=7)  # weekly seasonality
    
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))
    
    dates = daily_load.index
    fig.add_trace(go.Scatter(x=dates, y=daily_load, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=800, title_text="Daily Grid Load Decomposition")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("Decomposition uses **daily averages** with a 7‑day seasonal period (weekly pattern).")

# ------------------------------
# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a Random Forest model trained on historical EV charging data.\n"
    "The scaler (`scaler.pkl`) standardizes the features before prediction.\n"
    "All advanced analytics are computed on the test set (20% of data)."
)