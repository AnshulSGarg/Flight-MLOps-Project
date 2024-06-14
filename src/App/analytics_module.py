import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def analysis():

    st.header('Analytics')
    st.header("Minimum Flight Duration")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\min_flight_duration.png')
    st.header("Average Price by Trip Type")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_trip_type.png')
    st.header("Average Price by Carrier")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_carrier.png')
    st.header("Average Price by Days to Fly")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\days_to_fly_vs_price.png')
    st.header("Average Price by Round Trip Duration Days")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_round_trip_duration.png')
    st.header("Minimum Flight Duration")
    st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_weekday.png')

if __name__ == '__main__':
    analysis()
