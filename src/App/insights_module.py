import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pathlib

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
raw_path = home_dir.as_posix() 
insights_plot_path = raw_path  + r'/src/visualization/insights_plots/'

one_way_features = ['carrier', 'Airport_Route', 'stop','Days_to_Fly', 'from_hour', 'flight_duration_value', 'Holiday', 'Fly_WeekDay']

round_trip_features = ['carrier', 'Airport_Route', 'stop', 'Days_to_Fly', 'from_hour', 'flight_duration_value', 'Holiday', 'Fly_WeekDay','round_trip_duration']

option_list_one_way = ['Carrier','Airport Route','Layover','Days to Fly','Flight Departure Time','Flight Duration','Holiday','Day of Week']

option_list_round_trip = ['Carrier','Airport Route','Layover','Days to Fly','Flight Departure Time','Flight Duration','Holiday','Day of Week','Round Trip Duration']

description = [""" 
             - Spirit offers the lowest prices, whereas Delta has the highest prices. 
             - This suggests that selecting Spirit for one-way flights could result in substantial savings compared to choosing Delta.
             """,
             """ 
             - User can choose EWR as the preferred airport in New York if possible to get the lowest price. 
             - LAX to JFK seems to be most expensive options
             """,
             """ 
             - Non stop flight are more expensive compared to one with stops.
             - Having 1 or 2 stops doesn't make much difference from price point of view. 
             """,
             """ 
             - If user makes a booking to fly within a week, the flight rates are gonna be expensive.
             - Making a booking to fly between 30 to 60 days can be best time to get the best deal.
             """,
             """ 
             - There is hardly 5% price fluctuation, user can prefer and departure time for inbound flight.
             """,
             """ 
             - Flight price peaks between 5 - 6 hours duration, which is also shortest time.
             - If users are flexible, choosing a flight with longer flight duration can save some money.
             """,
             """ 
             - If user books a flight on a holiday price can be $30 higher compared to a regular day.
             """,
             """ 
             - Prices are higher if user fly on Sunday and lower for Tuesdays
             """,
             """ 
             - Booking a round trip for less than 30 days can be higher compared to returning after 30+ days.
             """]

def insights():
    st.header('Insights')

    st.write("""Here is insight on different features that impact price and can guide users in flight selection.  
            Select different feature labels to gain more insights.  
            """)

    selected = option_menu(
    menu_title = None,
    options = option_list_round_trip,
    icons = ["1_circle","1_circle","1_circle","1_circle","1_circle","1_circle","1_circle","1_circle","1_circle"],
    orientation = "horizontal")


    col1, col2 = st.columns([1, 1]) 

    with col1:
        if selected in(option_list_round_trip):
            idx = option_list_round_trip.index(selected)
            st.markdown("""
            <div style='text-align: center;'>
            <h5 style='margin: 0; padding: 0;'> Round Trip </h5>

            </div>
            """, unsafe_allow_html=True)
            feature = round_trip_features[idx]
            st.image(insights_plot_path + f'round_trip_{feature}.png')

    with col2:
        if selected in (option_list_one_way):
            idx = option_list_one_way.index(selected)
            st.markdown("""
            <div style='text-align: center;'>
            <h5 style='margin: 0; padding: 0;'> One Way </h5>
            </div>
            """, unsafe_allow_html=True)
            feature = one_way_features[idx]
            st.image(insights_plot_path + f'one_way_{feature}.png')
        else:
            pass

    st.write(description[idx])
    
if __name__ == '__main__':
    insights()

