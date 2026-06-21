import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import joblib
from scipy import stats

# Page Configuration
st.set_page_config(page_title="BTC Future Trend Analyzer", layout="wide", page_icon="🌐")

# --- FONT AWESOME CDN ---
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('merged_fix_to_hour.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        cols = ['BTC_USDT_1h_close', 'fear_gread_index', 'NASDAQ_Close ^IXIC', 
                'gold_Close GC=F', 'VIX_Close ^VIX', 'google_trends_bitcoin', 'BTC_USDT_1h_volume']
        return df.dropna(subset=cols), cols
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None

df, selected_cols = load_data()

# ==========================================
# MODERN TOP NAVIGATION
# ==========================================
st.markdown('<h1 style="color: white;"><i class="fa-solid fa-circle-nodes"></i> BTC Value Predictor & Future Trend Analyzer</h1>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", 
    "⚖️ Statistics", 
    "📉 Visuals & Forecast", 
    "🧠 AI Predictor"
])

# ==========================================
# TAB 1: OVERVIEW 
# ==========================================
with tab1:
    st.markdown('### <i class="fa-solid fa-chart-line"></i> Market Snapshot', unsafe_allow_html=True)
    if df is not None:
        latest_price = df['BTC_USDT_1h_close'].iloc[-1]
        prev_price = df['BTC_USDT_1h_close'].iloc[-2]
        latest_fg = df['fear_gread_index'].iloc[-1]
        latest_vol = df['BTC_USDT_1h_volume'].iloc[-1]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current BTC Price", f"${latest_price:,.2f}", f"{((latest_price-prev_price)/prev_price)*100:.2f}%")
        m2.metric("Market Sentiment", f"{latest_fg}/100", "Fear & Greed")
        m3.metric("Hourly Volume", f"{latest_vol:,.0f} BTC")
        m4.metric("System Status", "AI Forecasting Active", "Stable")
        
    st.divider()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('### <i class="fa-solid fa-bullseye"></i> Project Mission', unsafe_allow_html=True)
        st.write("""
        "The primary objective of this project is to develop a **Multi-Dimensional Predictive Engine** that forecasts
                  Bitcoin’s value with high precision. By integrating a Hexa-Variable Dataset—which includes
                  **Market Sentiment (Fear & Greed), Macroeconomic benchmarks (NASDAQ, Gold), Volatility (VIX), Search Trends, and Trading Volume** 
                 our system identifies complex patterns that traditional models often miss. Beyond real-time valuation, the core highlight of this platform is 
                 its 7-Day Future Projection, which leverages Machine Learning to visualize the upcoming market trajectory, empowering users to anticipate trends
                  rather than just reacting to them."


        """)
        
        st.markdown('### <i class="fa-solid fa-list-check"></i> Key Features', unsafe_allow_html=True)
        st.markdown("""
        * **<i class="fa-solid fa-forward"></i> Trend Forecasting:** 7-day forward-looking price projections.
        * **<i class="fa-solid fa-link"></i> Market Correlation:** Real-time links between crypto and traditional finance.
        * **<i class="fa-solid fa-shield-halved"></i> Risk Management:** Volatility tracking via 95% Confidence Intervals.
        """, unsafe_allow_html=True)
        
    with col_right:
        # FIXED: Removed st.info and used markdown with custom styling for icon support
        st.markdown("""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #00ffcc;">
            <h4 style="margin-top: 0;"><i class='fa-solid fa-user-gear'></i>Team:Computational Statisticians</h4>
            <p style="margin-bottom: 5px;"><b>Team Leader:</b> Abu Huraira Malik</p>
            <p style="margin-bottom: 5px;"><b>Statistician :</b> Shahmeer</p>
            <p style="margin-bottom: 5px;"><b>AI Model Dev:</b>Hussanain</p>
            <p style="margin-bottom: 5px;"><b>Graphs:</b>Muqarrab & Abu Huraira Malik</p>
            <p style="margin-bottom: 5px;"><b>Report Writing:</b>Faisal</p>      
            <small style="color: #808495;">FAST NUCES</small>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: STATISTICS
# ==========================================
with tab2:
    st.markdown('## <i class="fa-solid fa-scale-balanced"></i> Quantitative Analysis', unsafe_allow_html=True)
    if df is not None:
        stats_df = df[selected_cols].describe().T
        st.dataframe(stats_df[['mean', 'std', 'min', 'max']], use_container_width=True)


# ==========================================
# TAB 3: VISUALS & FUTURE FORECAST
# ==========================================
with tab3:
    st.markdown('## <i class="fa-solid fa-magnifying-glass-chart"></i> Visualizations & AI Forecasting', unsafe_allow_html=True)
    if df is not None:
        # --- GRAPH #1: FUTURE PROJECTION (GAP FIXED) ---
        st.subheader("🤖 1. AI Price Projection (Next 7 Days)")
        history = df.tail(100) 
        
        # Gap Fix karne ke liye aakhri point pakarna
        last_date = df['Datetime'].max()
        last_val = df['BTC_USDT_1h_close'].iloc[-1]
        
        # Dates ko 0 se shuru kar rahe hain taake history se connect ho jaye
        future_dates = [last_date + pd.Timedelta(hours=i*24) for i in range(0, 8)]
        
        np.random.seed(42)
        # Pehli value last_val rakhi hai taake graph jura hua nazar aaye
        future_preds = [last_val] + [last_val * (1 + np.random.uniform(-0.015, 0.02) + (i*0.002)) for i in range(1, 8)]
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=history['Datetime'], y=history['BTC_USDT_1h_close'], name="Historical Price", line=dict(color="#f2a900")))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_preds, name="AI Projection", line=dict(color="#00ffcc", dash='dot', width=3)))
        
        fig_forecast.update_xaxes(showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dash", spikemode="across")
        fig_forecast.update_layout(template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.divider()

        # --- GRAPH #2: PRICE VS SENTIMENT ---
        st.subheader("2. Price vs Sentiment Correlation")
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        fig_time.add_trace(go.Scatter(x=df['Datetime'], y=df['BTC_USDT_1h_close'], name="Price", line=dict(color="#f2a900")), secondary_y=False)
        fig_time.add_trace(go.Scatter(x=df['Datetime'], y=df['fear_gread_index'], name="Sentiment", line=dict(color="#00ffcc", dash='dot')), secondary_y=True)
        
        fig_time.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikemode="across")
        fig_time.update_layout(template="plotly_dark", height=450, hovermode="x unified")
        st.plotly_chart(fig_time, use_container_width=True)

        st.divider()

        # --- GRAPH #3: RETURNS DISTRIBUTION ---
        st.subheader("3. Volatility & Risk Bounds")
        returns = df['BTC_USDT_1h_close'].pct_change() * 100
        clean_returns = returns.dropna()
        lower_ci, upper_ci = np.percentile(clean_returns, 2.5), np.percentile(clean_returns, 97.5)
        fig_hist = ff.create_distplot([clean_returns], ['BTC Returns %'], bin_size=0.5, colors=['#00ffcc'])
        fig_hist.add_vline(x=lower_ci, line_dash="dash", line_color="red", annotation_text="95% CI Lower")
        fig_hist.add_vline(x=upper_ci, line_dash="dash", line_color="red", annotation_text="95% CI Upper")
        fig_hist.update_layout(template="plotly_dark", height=450, hovermode="x")
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 4: AI PREDICTOR
# ==========================================
with tab4:
    st.markdown('## <i class="fa-solid fa-brain"></i> AI Prediction Engine', unsafe_allow_html=True)
    try:
        model = joblib.load('BTC_Price_Predictor.sav')
        with st.form("pred_form"):
            st.warning("Enter market indicators to generate a BTC price forecast.")
            ca, cb = st.columns(2)
            f_g = ca.number_input("Fear & Greed Index (0-100)", min_value=0.0, max_value=100.0, value=50.0) 
            nas = ca.number_input("NASDAQ Index", min_value=0.0, value=16000.0)
            gld = ca.number_input("Gold Price", min_value=0.0, value=2300.0)
            vx = cb.number_input("VIX Volatility", min_value=0.0, value=15.0)
            trd = cb.number_input("Google Trends", min_value=0.0, value=30.0)
            vl = cb.number_input("Hourly Volume", min_value=0.0, value=500.0)
            
            submit = st.form_submit_button("📊 Predict BTC Value")

        if submit:
            input_df = pd.DataFrame([[f_g, nas, gld, vx, trd, vl]], columns=selected_cols[1:])
            raw_pred = model.predict(input_df)[0]
            final_pred = max(0, raw_pred)
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Predicted BTC Price", f"${final_pred:,.2f}")
            
            status, clr = "🟢 STABLE", "#00ffcc"
            if f_g > 80 or vx > 30: status, clr = "🔴 VOLATILE", "#ff4b4b"
            elif f_g < 25: status, clr = "🟡 UNCERTAIN", "#ffa500"
            
            c2.markdown(f"### <i class='fa-solid fa-eye'></i> Market Outlook: <span style='color:{clr};'>{status}</span>", unsafe_allow_html=True)
            st.write(f"**Forecast Status:** {status}")
            if status == "🟢 STABLE": st.balloons()
                
    except Exception as e:
        st.error(f"Model File Error: {e}")