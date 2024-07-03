import streamlit as st
from price_prediction import Price_Prediction
from analytics_module import analysis
from home_page import landing_page
from prediction_explanation import explain
from insights_module import insights


st.set_page_config(layout="wide")  # Set the layout to wide mode

# Page 1
def analysis_page():
    analysis()

# Page 2
def Prediction():
    Price_Prediction()

# Page 3
def insight_page():
    insights()

# Page 4
def understanding_prediction():
    explain()

# Sidebar navigation
page_options = {
    "Home": landing_page,
    "Analytics": analysis_page,
    "Price Prediction": Price_Prediction,
    "Understand Price Prediction": understanding_prediction,
    "Insights": insight_page
}

# Sidebar to select pages
selected_page = st.sidebar.radio("Select Page", list(page_options.keys()))

# Display selected page
page_options[selected_page]()
