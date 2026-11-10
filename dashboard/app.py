"""
NYC Taxi Demand Forecasting - interactive story dashboard.

Run from this folder:
    streamlit run app.py
"""

import streamlit as st

from utils import apply_style, check_data_built

st.set_page_config(
    page_title="NYC Taxi Demand | Team 12",
    page_icon=":material/local_taxi:",
    layout="wide",
)
apply_style()
check_data_built()

pages = {
    "Overview": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "The story": [
        st.Page("views/ch1_data.py", title="1 · Data collection & cleaning", icon=":material/database:"),
        st.Page("views/ch2_eda.py", title="2 · Exploring demand", icon=":material/insights:"),
        st.Page("views/ch3_models.py", title="3 · Predicting demand", icon=":material/model_training:"),
        st.Page("views/ch4_time_series.py", title="4 · Forecasting ahead", icon=":material/timeline:"),
        st.Page("views/ch5_rules.py", title="5 · Demand rules", icon=":material/account_tree:"),
        st.Page("views/ch6_impact.py", title="6 · Business impact", icon=":material/flag:"),
    ],
    "Try it live": [
        st.Page("views/tool_predictor.py", title="Demand predictor", icon=":material/online_prediction:"),
        st.Page("views/tool_forecast.py", title="Forecast planner", icon=":material/calendar_month:"),
        st.Page("views/tool_rules.py", title="Dispatch advisor", icon=":material/alt_route:"),
    ],
}

page = st.navigation(pages)

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="bar"></div>'
        '<div class="name">NYC Taxi Demand</div>'
        '<div class="sub">Weather-Aware Taxi Demand Prediction and Forecasting in NYC<br>'
        '23CSE452 Business Analytics · Team 12</div></div>',
        unsafe_allow_html=True,
    )

page.run()
