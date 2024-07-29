import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def landing_page():    
    st.markdown("""
    <div style='text-align: center;'>
        <h5 style='margin: 0; padding: 0;'>Welcome to</h5>
        <h1 style='color: #1e90ff; margin: 0; padding: 0;'>JetPredict</h1>
        <h4 style='color: #6fa8dc; margin: 0; padding: 0;'>Smart Predictions and Travel Insights App</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1]) 

    with col1:
        st.text(" \n")
        st.text(" \n")
        st.write("""
            ##### _Description_
            JetPredict is the ultimate platform for predicting flight fares, focusing on route between airports in New York and Los Angeles.. 
            There are 4 different pages that can help user make flight fare prediction, understand contribution of different users selections to the prediction, learn about impact of different selection on flight fare from analysis and insights.
                 """)

    with col2:
        def analysis():
            data = {
                'airport': ['Los Angeles', 'New York'],
                'latitude': [34.0522, 40.7128],
                'longitude': [-118.2437, -74.0060]
            }

            map_df = pd.DataFrame(data)

            highlight_df = map_df.copy()

            highlight_df.loc[highlight_df['airport'] == 'Los Angeles', 'marker_size'] = 7
            highlight_df.loc[highlight_df['airport'] == 'New York', 'marker_size'] = 7

            highlight_df['marker_size'] = highlight_df['marker_size'].fillna(5)

            center_lat = (34.0522 + 40.7128) / 2
            center_lon = (-118.2437 + -74.0060) / 2

            fig = px.scatter_mapbox(highlight_df, lat="latitude", lon="longitude",
                                    color_discrete_sequence=["red"],
                                    size='marker_size', zoom=1.5,
                                    mapbox_style="open-street-map", width=20, height=300, hover_name="airport")
            fig.update_layout(mapbox=dict(center=dict(lat=center_lat, lon=center_lon)), showlegend=False)

            line_trace = go.Scattermapbox(
                lat=[34.0522, 40.7128],
                lon=[-118.2437, -74.0060],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='Flight Path'
            )

            fig.add_trace(line_trace)

            st.plotly_chart(fig, use_container_width=True)

        analysis()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("""
            ###### _Analytics Module_:
            - Understand what dataset tells about impact of user selections on flight price.
            - Explore pricing variations and other factors influencing fares through advanced analytics charts.""")
        
        st.text(" \n")

        st.write("""
        ###### _Understand Price Prediction Module_:
        - Learn the contribution made by different user selections on the predicted prices.
        - Get an opportunity to play with different selections and understand more about predicted price.""")

    with col2:

        st.write("""
            ###### _Prediction Module_:
            - Make flight fare predictions to help you plan your trip and know beforehand estimate price.
            - Understand prediction trend and identify optimal booking windows for the best deals.""")
        
        st.text(" \n")

        st.write("""
        ###### _Insights Module_:
        - Explore detailed insights derived from predictive model. 
        - Understand fare fluctuations due to different user selections values.""")
        

                        
    st.text(" \n")
    st.text(" \n")
    st.write("""
            Whether you're a frequent traveler, budget-conscious adventurer, or data enthusiast, JetPredict offers tools to meet your needs. 
             Experience the future of flight fare prediction and analysis with JetPredict today!""")
    

    st.text(" \n")
    st.write("""
    This project stemmed from my passion for data science and my desire to apply my knowledge in a practical and impactful manner.  
             If you have any questions please reach out www.linkedin.com/AnshulSGarg  
             Checkout my other projects on github https://github.com/AnshulSGarg """)

