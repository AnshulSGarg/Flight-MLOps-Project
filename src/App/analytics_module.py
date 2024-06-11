import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

def analysis():
    st.header('Analytics')

    data = {
        'airport': ['Los Angeles', 'New York'],
        'latitude': [34.0522, 40.7128],
        'longitude': [-118.2437, -74.0060]
    }

    map_df = pd.DataFrame(data)

    # Separate DataFrame for highlighting specific airports
    highlight_df = map_df.copy()

    # Set marker size for Los Angeles and New York
    highlight_df.loc[highlight_df['airport'] == 'Los Angeles', 'marker_size'] = 7  # Smaller size for Los Angeles
    highlight_df.loc[highlight_df['airport'] == 'New York', 'marker_size'] = 7  # Smaller size for New York

    # Set default marker size for other airports
    highlight_df['marker_size'].fillna(5, inplace=True)

    fig = px.scatter_mapbox(highlight_df, lat="latitude", lon="longitude",
                            color_discrete_sequence=["red"],
                            size='marker_size', zoom=3,
                            mapbox_style="open-street-map", width=450, height=450, hover_name="airport")  # Adjust width and height

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

    st.text('Hello')

# Run the analysis function
analysis()
