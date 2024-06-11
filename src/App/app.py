import streamlit as st
from price_prediction import Price_Prediction
from analytics_module import analysis
from home_page import landing_page


st.set_page_config(layout="wide")  # Set the layout to wide mode

# Page 1
def Prediction():
    Price_Prediction()

# Page 2
def page_two():
    analysis()

# Page 3
def page_three():
    st.title("Page Three")
    st.write("This is Page Three.")

# Page 4
def page_four():
    st.title("Page Four")
    st.write("This is Page Four.")

# Sidebar navigation
page_options = {
    "Home": landing_page,
    "Analytics": page_two,
    "Price Prediction": Price_Prediction,
    "Insight": page_three,
    "Recommendation": page_four
}

# Sidebar to select pages
selected_page = st.sidebar.radio("Select Page", list(page_options.keys()))

# Display selected page
page_options[selected_page]()
