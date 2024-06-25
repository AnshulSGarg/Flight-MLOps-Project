import streamlit as st
import matplotlib.pyplot as plt

one_way_features = ['carrier', 'Airport_Route', 'stop',
       'Days_to_Fly', 'from_hour',
       'flight_duration_value', 'Holiday', 'Fly_WeekDay']

round_trip_features = ['carrier', 'Airport_Route', 'stop',
       'Days_to_Fly', 'from_hour',
       'flight_duration_value', 'Holiday', 'Fly_WeekDay','round_trip_duration']

def insights():
    st.header('Insights')
    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.header('One Way')
        for feature in one_way_features:
            st.subheader(feature)
            st.image(fr'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\insights_plots\one_way_{feature}.png')
    
    with col2:
        st.header('Round Trip')
        for feature in round_trip_features:
            st.subheader(' ')
            st.image(fr'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\insights_plots\round_trip_{feature}.png')

if __name__ == '__main__':
    insights()

