import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from io import StringIO

st.set_page_config(page_title="QuantifyAI v0.2.1", layout="wide")
st.title("🚀 QuantifyAI v0.2.1")
st.caption("AI-Enhanced Quantification Tool for Malaria Commodities | Focused on ACTs (Malawi-style)")

# ------------------- DEFAULT SAMPLE DATA -------------------
@st.cache_data
def load_default_data():
    data = """date,district,product_code,product_name,consumption_qty,stock_on_hand,shipments_received,adjustments,rainfall_mm,reported_cases
2025-01,District A,MC-001,ACT 6x4 Tablets,450,1250,600,-20,152.3,420
2025-01,District A,MC-002,RDT Kits,320,850,450,10,152.3,420
2025-01,District B,MC-001,ACT 6x4 Tablets,380,1050,550,-10,138.7,380
2025-01,District B,MC-002,RDT Kits,250,720,380,5,138.7,380
2025-02,District A,MC-001,ACT 6x4 Tablets,480,1180,500,30,120.5,450
2025-02,District A,MC-002,RDT Kits,340,780,300,-15,120.5,450
2025-02,District B,MC-001,ACT 6x4 Tablets,410,980,450,20,115.2,400
2025-02,District B,MC-002,RDT Kits,270,680,250,10,115.2,400
2025-03,District A,MC-001,ACT 6x4 Tablets,520,1050,400,-10,85.6,480
2025-03,District A,MC-002,RDT Kits,360,720,350,5,85.6,480
2025-03,District B,MC-001,ACT 6x4 Tablets,440,920,380,-25,82.4,430
2025-03,District B,MC-002,RDT Kits,290,630,220,-5,82.4,430
2025-04,District A,MC-001,ACT 6x4 Tablets,500,980,450,10,65.8,460
2025-04,District A,MC-002,RDT Kits,350,680,280,15,65.8,460
2025-04,District B,MC-001,ACT 6x4 Tablets,420,860,350,5,62.1,410
2025-04,District B,MC-002,RDT Kits,280,590,210,20,62.1,410"""
    df = pd.read_csv(StringIO(data))
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_default_data()

# ------------------- UPLOAD -------------------
st.sidebar.header("📤 Upload Your Real Data (36 months)")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        st.sidebar.success(f"✅ Loaded {uploaded_file.name} — {len(df)} rows")
    except Exception as e:
        st.sidebar.error(f"Upload error: {str(e)}")

# ------------------- FILTERS -------------------
products = df['product_name'].unique()
selected_products = st.sidebar.multiselect("Select Products", products, default=["ACT 6x4 Tablets"])

view_level = st.sidebar.radio("View Level", ["National (Aggregated)", "By District"])

# ------------------- DATA PREPARATION -------------------
if view_level == "National (Aggregated)":
    working_df = df.groupby(['date', 'product_name']).agg({
        'consumption_qty': 'sum',
        'stock_on_hand': 'sum',
        'shipments_received': 'sum',
        'adjustments': 'sum',
        'rainfall_mm': 'mean',
        'reported_cases': 'sum'
    }).reset_index()
else:
    working_df = df.copy()   # Keep district level

# ------------------- 1. DATA OVERVIEW -------------------
st.header("1. Data Overview & AI Intelligence")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data Preview")
    display_df = working_df[working_df['product_name'].isin(selected_products)]
    st.dataframe(display_df.head(10), use_container_width=True)

with col2:
    st.subheader("AI Anomaly Detection")
    for prod in selected_products:
        sub = working_df[working_df['product_name'] == prod]
        if len(sub) < 8: 
            continue
        model = IsolationForest(contamination=0.1, random_state=42)
        sub = sub.copy()
        sub['anomaly'] = model.fit_predict(sub[['consumption_qty']].values)
        anomalies = sub[sub['anomaly'] == -1]
        if not anomalies.empty:
            st.warning(f"⚠️ {prod}: {len(anomalies)} anomalies")

# ------------------- 2. FORECASTING -------------------
st.header("2. AI Forecasting Engine (QAT + AI Upgrade)")
horizon = st.slider("Forecast Horizon (months)", 6, 36, 24)

tab1, tab2 = st.tabs(["Consumption Forecast (ACTs)", "Stock Status Matrix (QAT-style)"])

with tab1:
    for prod in selected_products:
        st.subheader(f"Forecast for {prod}")
        
        # Aggregate to monthly per product (critical fix for By District)
        if view_level == "By District":
            sub = working_df[working_df['product_name'] == prod].groupby('date').agg({
                'consumption_qty': 'sum',
                'rainfall_mm': 'mean',
                'reported_cases': 'sum'
            }).reset_index()
        else:
            sub = working_df[working_df['product_name'] == prod].copy()
        
        if len(sub) < 8:
            st.warning("Not enough data for reliable forecast yet.")
            continue
        
        sub = sub.sort_values('date').set_index('date')
        cons_series = sub['consumption_qty'].asfreq('MS').fillna(method='ffill')
        
        # ARIMA
        try:
            arima_model = ARIMA(cons_series, order=(1,1,1)).fit()
            arima_fc = arima_model.forecast(horizon)
        except:
            arima_fc = pd.Series([cons_series.iloc[-1]] * horizon, 
                               index=pd.date_range(cons_series.index[-1], periods=horizon, freq='MS'))
        
        # Prophet with covariates
        prophet_data = sub.reset_index()[['date', 'consumption_qty', 'rainfall_mm', 'reported_cases']]
        prophet_data = prophet_data.rename(columns={'date': 'ds', 'consumption_qty': 'y'})
        m = Prophet(yearly_seasonality=True)
        m.add_regressor('rainfall_mm')
        m.add_regressor('reported_cases')
        m.fit(prophet_data)
        future = m.make_future_dataframe(periods=horizon, freq='MS')
        future['rainfall_mm'] = prophet_data['rainfall_mm'].iloc[-1]
        future['reported_cases'] = prophet_data['reported_cases'].iloc[-1] * 1.05
        prophet_fc = m.predict(future)
        
        ensemble_fc = (arima_fc.values + prophet_fc['yhat'].iloc[-horizon:].values) / 2
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cons_series.index, y=cons_series.values, name="Historical", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=arima_fc.index, y=arima_fc.values, name="ARIMA", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=prophet_fc['ds'].iloc[-horizon:], y=prophet_fc['yhat'].iloc[-horizon:], name="AI Prophet", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=future['ds'].iloc[-horizon:], y=ensemble_fc, name="AI Ensemble", line=dict(color="red", width=3)))
        fig.update_layout(title=f"{prod} - {view_level} Forecast", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric(f"Projected Need (next 12 months)", f"{int(ensemble_fc[:12].sum()):,}")

with tab2:
    st.subheader("Stock Status Matrix")
    latest = working_df.groupby('product_name').last().reset_index()
    matrix = latest[['product_name', 'stock_on_hand', 'consumption_qty']].copy()
    matrix['AMC'] = matrix['consumption_qty']
    matrix['MOS'] = (matrix['stock_on_hand'] / matrix['AMC']).round(1)
    st.dataframe(matrix, use_container_width=True)

st.caption("QuantifyAI v0.2.1 | Fixed district-level handling")
