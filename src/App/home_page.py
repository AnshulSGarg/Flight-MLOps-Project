import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def landing_page():    
    st.markdown("""
    <div style='text-align: center;'>
        <h5 style='margin: 0; padding: 0; text-align: left; margin-left: 430px;'>Welcome to</h5>
        <h1 style='color: #1e90ff; margin: 0; padding: 0;'>JetPredict</h1>
        <h4 style='color: #6fa8dc; margin: 0; padding: 0;'>Smart Predictions and Travel Insights App</h4>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for layout with equal width
    col1, col2 = st.columns([1, 1])  # Adjust the width ratio to equal columns

    with col1:
        st.write("""
            ##### Description
            JetPredict is the ultimate platform for predicting flight fares, initially targeting New York to Los Angeles routes while exploring possibilities for expansion to other destinations. 
            This project stemmed from my passion for data science and my desire to apply my knowledge in a practical and impactful manner.

            ###### Analytics Module:
            Gain insights into historical flight data trends and patterns. Explore pricing variations and other factors influencing fares through advanced analytics tools.

            ###### Prediction Module:
            Predict future flight fares accurately using machine learning algorithms. Consider factors like departure time, airline, and historical data to plan your travel budget.

            ###### Insight Module:
            Explore detailed analysis derived from predictive models. Understand fare fluctuations and identify optimal booking windows for the best deals.

            ###### Recommendation Module:
            Receive personalized flight recommendations based on your preferences and budget. Utilize machine learning techniques to discover cost-effective travel options.

            Whether you're a frequent traveler, budget-conscious adventurer, or data enthusiast, JetPredict offers tools to meet your needs. Experience the future of flight fare prediction and analysis with JetPredict today!""")

    with col2:
        def analysis():
            data = {
                'airport': ['Los Angeles', 'New York'],
                'latitude': [34.0522, 40.7128],
                'longitude': [-118.2437, -74.0060]
            }

            map_df = pd.DataFrame(data)

            # Separate DataFrame for highlighting specific airports
            highlight_df = map_df.copy()

            # Set marker size for Los Angeles and New York
            highlight_df.loc[highlight_df['airport'] == 'Los Angeles', 'marker_size'] = 7
            highlight_df.loc[highlight_df['airport'] == 'New York', 'marker_size'] = 7

            # Set default marker size for other airports
            highlight_df['marker_size'].fillna(5, inplace=True)

            # Center the map and adjust the zoom to fit all points
            center_lat = (34.0522 + 40.7128) / 2
            center_lon = (-118.2437 + -74.0060) / 2

            fig = px.scatter_mapbox(highlight_df, lat="latitude", lon="longitude",
                                    color_discrete_sequence=["red"],
                                    size='marker_size', zoom=2.5,  # Adjust zoom level
                                    mapbox_style="open-street-map", width=450, height=450, hover_name="airport")  # Adjust width and height
            fig.update_layout(mapbox=dict(center=dict(lat=center_lat, lon=center_lon)),
                              showlegend=False)  # Hide the legend

            # Add a line connecting Los Angeles and New York
            line_trace = go.Scattermapbox(
                lat=[34.0522, 40.7128],
                lon=[-118.2437, -74.0060],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='Flight Path'
            )

            # Add the line trace to the figure
            fig.add_trace(line_trace)

            st.plotly_chart(fig, use_container_width=True)

        # Run the analysis function
        analysis()