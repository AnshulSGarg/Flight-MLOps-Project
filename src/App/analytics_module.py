import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def analysis():

    st.header('Analytics')

    # st.markdown("""
    # <div style='text-align: left;'>
    # <h6 style='margin: 0; padding: 0;'> - Here is detailed analysis of impact of different features selection of flight booking on price. </h6>
    # <h6 style='margin: 0; padding: 0;'> - Select different feature labels to see how they are related to price of the flight. </h6>
    # <h6 style='margin: 0; padding: 0;'> </h6>         
    # </div>
    # """, unsafe_allow_html=True)

    st.write("""Here is a detailed analysis of the impact of different feature selections on flight booking prices.  
             Select different feature labels to see how they are related to the flight price.  
             
             """)

    selected = option_menu(
        menu_title = None,
        options = ["Flight Duration","Trip Type","Carrier","Days to Fly","Round Trip Duration","Day of the week"],
        icons = ["1_circle","1_circle","1_circle","1_circle","1_circle","1_circle"],
        orientation = "horizontal"
    )

    if selected == "Flight Duration":
        # st.subheader("Minimum Flight Duration")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\min_flight_duration.png')
        st.write("""
                - Shorted flight duration is 5 hours for route between Los Angeles International Airport and Newark Liberty International Airport
                - For LA to NY Route, shorted flight duration is 5 hours 54 mins for route between Los Angeles International Airport and La Guardia Airport
                - Whereas, for NY to LA Route, shorted flight duration is 5 hours 47 mins for route between John F Kennedy International Airport and Los Angeles International Airport
                """)    
        
    if selected == "Trip Type":
        # st.subheader("Average Price by Trip Type")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_trip_type.png')
        file_path = r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\carrier_overhead_bin.csv'
        overhead_bin_df = pd.read_csv(file_path)
        overhead_bin_df.rename(columns={'carrier':'Carrier','overhead_bin': 'Overhead Bin'}, inplace=True)
        st.write("""
        - Spirit airlines has lowest average price.
        - Delta and American airlines have highest average price.
        - Carriers like JetBlue, Spirit and United charge additional cost for overhead bins which can be around \$35-\$45
        - For one way trip it can add \$35 and for round trip around \$70, making carrier like JetBlue price almost same as Delta and American airlines.
            """)
        st.dataframe(overhead_bin_df[['Carrier','Overhead Bin']], hide_index=True)

    if selected == "Carrier":
        # st.subheader("Average Price by Carrier")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_carrier.png')
        st.write("""
        - Average price for route LA to NY around 5% more than average price for route NY to LA
        - Looking at the boxplot, there is lot of variation in price for each carrier. Price variation for delta are comparatively more stable. 
        - This highlights that price can be extremely higher compared to average
            """)

    if selected == "Days to Fly":
        # st.subheader("Average Price by Days to Fly")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\days_to_fly_vs_price.png')
        st.write("""
        - Days to Fly is the difference in days between flight operation day and day of booking the flight.
        - Average price are very high when making booking couple of days before day of flying.
        - Average price starts dropping as the difference in days increases.
        - However, it can also be observed that if the booking is made 3 months in advance it can be bit more expensive compared to booking it 1 month or 2 months in advance.
        - You can know more about impact of days to fly on flight price by clicking on other tabs like Price Prediction and Insight
            """)

    if selected == "Round Trip Duration":
        # st.subheader("Average Price by Round Trip Duration Days")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_round_trip_duration.png')
        st.write("""
        - Round Trip Duration is number of days between inbound and outbound flights
        - There seems to be no pattern between Price and Round Trip Days, highlighting that this feature might not impact price much.
        - Understanding Prediction Page can give more details information on this features based on custom user selection.
            """)
    if selected == "Day of the week":
        # st.subheader("Average price by Weekday")
        st.image(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\src\visualization\analytics_plots\price_by_weekday.png')
        st.write("""
        - Average price is higher for Sunday, followed by Thursday and Monday. 
        - Booking a flight for Tuesday and Wednesday can be comparatively cheaper.                  
            """)

if __name__ == '__main__':
    analysis()
