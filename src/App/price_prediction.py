import pandas as pd
import streamlit as st
import altair as alt
import pickle
import datetime
import numpy as np
import pathlib

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
raw_path = home_dir.as_posix() 
pickle_path = raw_path  + r'/pickle_files/'

def Price_Prediction():
    st.header("Price Prediction")
    st.subheader("Select Input")

    with open(pickle_path + 'flight_df.pkl','rb') as file:
        df = pickle.load(file)

    with open(pickle_path + 'flight_pipeline.pkl','rb') as file:
        pipeline = pickle.load(file)

    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2024-01-01', end='2024-12-31').to_pydatetime()
    holiday_df = pd.DataFrame(holidays, columns=['Holiday'])

    col1, col2, col3 = st.columns([1.7, 1, 1])

    with col1:
        From_Airport = st.selectbox('From Airport',['Los Angeles International Airport (LAX)','Newark International Airport (EWR)','John F Kennedy International Airport (JFK)','La Guardia Airport (LGA)'])
        if From_Airport == 'Los Angeles International Airport (LAX)':
            From_Airport_Code = 'LAX'
        elif From_Airport == 'Newark International Airport (EWR)':
            From_Airport_Code = 'EWR'
        elif From_Airport == 'John F Kennedy International Airport (JFK)':
            From_Airport_Code = 'JFK'
        elif From_Airport == 'La Guardia Airport (LGA)':
            From_Airport_Code = 'LGA'

        if From_Airport == 'Los Angeles International Airport (LAX)':
            To_Airport = st.selectbox('To Airport',['Newark International Airport (EWR)','John F Kennedy International Airport (JFK)','La Guardia Airport (LGA)'])
        else:
            To_Airport = st.selectbox('To Airport', ['Los Angeles International Airport (LAX)'])
        if To_Airport == 'Los Angeles International Airport (LAX)':
            To_Airport_Code = 'LAX'
        elif To_Airport == 'Newark International Airport (EWR)':
            To_Airport_Code = 'EWR'
        elif To_Airport == 'John F Kennedy International Airport (JFK)':
            To_Airport_Code = 'JFK'
        elif To_Airport == 'La Guardia Airport (LGA)':
            To_Airport_Code = 'LGA'

        Trip_Type = st.selectbox('Trip_Type', ['One Way','Rounds Trip'])

    with col3:
        carrier = st.selectbox('Carrier', df['carrier'].unique().tolist())
        

        def float_to_time(hour_float):
            hour = int(hour_float)
            minute = int((hour_float - hour) * 60)
            return f"{hour:02d}:{minute:02d}"
        
        time_options = [float_to_time(i/4) for i in range(0, 96)]

        from_hour = st.selectbox('from_hour', time_options, index=27)
        flight_duration_value = st.selectbox('Flight Duration (hour)',[float(i/2) for i in range(10, 25)])

    with col2:
        from_date = st.selectbox('Start Date',[datetime.date.today() + datetime.timedelta(days=1) + datetime.timedelta(days=i) for i in range(90)])
        if Trip_Type == 'Rounds Trip':
            to_date = st.selectbox('Return Date', [from_date + datetime.timedelta(days=1) + datetime.timedelta(days=i) for i in range(90)])
        else:
            to_date = st.selectbox('Return Date',[''])
        if from_date in holiday_df:
            Holiday = 'Holiday'
        else:
            Holiday = 'Not_Holiday'

        stop = st.selectbox('Stops', ['Nonstop','1 stop','2 stops'])
        datetime_value = pd.to_datetime(from_date)
        Fly_WeekDay_Name = datetime_value.day_name()
        if Fly_WeekDay_Name == 'Monday':
            Fly_WeekDay = 1
        elif Fly_WeekDay_Name == 'Tuesday':
            Fly_WeekDay = 2
        elif Fly_WeekDay_Name == 'Wednesday':
            Fly_WeekDay = 3
        elif Fly_WeekDay_Name == 'Thursday':
            Fly_WeekDay = 4
        elif Fly_WeekDay_Name == 'Friday':
            Fly_WeekDay = 5
        elif Fly_WeekDay_Name == 'Saturday':
            Fly_WeekDay = 6
        elif Fly_WeekDay_Name == 'Sunday':
            Fly_WeekDay = 7
        if Trip_Type == 'Rounds Trip':
            round_trip_duration = (to_date - from_date).days
        else:
            round_trip_duration = 0
        Days_to_Fly = (from_date - datetime.date.today()).days




    Airport_Route = From_Airport_Code + ' - ' + To_Airport_Code
    data = [[carrier,Trip_Type,Airport_Route
             ,stop,round_trip_duration
             ,Days_to_Fly ,flight_duration_value,
              Holiday, Fly_WeekDay,from_hour]]
    columns = ['carrier', 'Trip_Type', 'Airport_Route',
           'stop', 'round_trip_duration',
           'Days_to_Fly', 'flight_duration_value',
            'Holiday', 'Fly_WeekDay' ,'from_hour']

    one_df = pd.DataFrame(data, columns=columns)

    # print(one_df)
    # st.dataframe(one_df)


    # Number of new rows to add
    num_new_rows = 90

    # Generate new rows with increasing values for Days_to_Fly
    new_rows = []
    for i in range(1, num_new_rows + 1):
        new_row = one_df.iloc[0].copy()  # Copy existing row
        new_row['Days_to_Fly'] = i
        new_rows.append(new_row)

    trend_df = one_df
    # Append new rows to the DataFrame
    trend_df = pd.concat([trend_df, pd.DataFrame(new_rows)], ignore_index=True)
    trend_df = trend_df.drop(trend_df.index[0])
    # st.dataframe(trend_df)
    with open(pickle_path + 'one_df.pkl', 'wb') as f:
        pickle.dump(one_df, f)

    if st.button('Predict Price'):
    # Make a prediction
        prediction = pipeline.predict(one_df)
        prediction = np.expm1(prediction)
        with open(pickle_path + 'predicted_price.pkl', 'wb') as f:
            pickle.dump(prediction, f)
        lower_bound = round(prediction[0]) - 30
        upper_bound = round(prediction[0]) + 30
        # st.text('Predicted price should be between ${} and ${}'.format(lower_bound, upper_bound))
        st.markdown(f"""
        <div style='text-align: left;'>
        <h6 style='margin: 0; padding: 0;'> </h6> 
        <h5 style='margin: 0; padding: 0;'> Predicted price should be between ${lower_bound} and ${upper_bound} </h5>
        <h6 style='margin: 0; padding: 0;'> </h6>       
        <h6 style='margin: 0; padding: 0;'> </h6> 
        <h6 style='margin: 0; padding: 0;'> </h6>   
        </div>
        """, unsafe_allow_html=True)

        st.title("Price Trend")
        # Predict the trend
        pred_array = pipeline.predict(trend_df)
        pred_array = np.round(np.expm1(pred_array), 0)
        # print(len(pred_array))

        # Get the current date
        current_date = datetime.datetime.now().date()

        # Number of new rows
        num_new_rows = 90

        # Create a list of dates starting from the current date
        date_range = [current_date + datetime.timedelta(days=i) for i in range(num_new_rows)]

        # Create chart data
        chart_data = {
            'x': date_range,  # Use datetime objects for sorting
            'Price': pred_array
        }

        # Create DataFrame
        chart_df = pd.DataFrame(chart_data)

        # Ensure 'x' column is in datetime format
        chart_df['x'] = pd.to_datetime(chart_df['x'])

        # Sort the DataFrame by date in ascending order
        chart_df = chart_df.sort_values('x')

        # Create a string column for display purposes
        chart_df['x_str'] = chart_df['x'].dt.strftime('%b %d')

        # Specify the point to highlight
        highlight_index = chart_df[chart_df['x'] == pd.to_datetime(from_date)].index
        highlight_point = chart_df.iloc[highlight_index]

        # Create the base line chart with points
        line_chart = alt.Chart(chart_df).mark_line().encode(
            x=alt.X('x:T', axis=alt.Axis(labelAngle=-45, title='Date', format='%b %d')),
            y=alt.Y('Price', axis=alt.Axis(format='$,.0f', title='Price')),  # Format y-axis as dollars without decimal
            tooltip=[alt.Tooltip('x_str', title='Date'), alt.Tooltip('Price', format='$,.0f')],  # Add tooltips for both x_str and y values
            color=alt.value('#ADD8E6')  # Set the color to light blue
        ).interactive()

        point_chart = alt.Chart(chart_df).mark_circle(size=50).encode(
            x=alt.X('x:T'),
            y=alt.Y('Price', axis=alt.Axis(format='$,.0f')),  # Format y-axis as dollars without decimal
            tooltip=[alt.Tooltip('x_str', title='Date'), alt.Tooltip('Price', format='$,.0f')],  # Add tooltips for both x_str and y values
            color=alt.value('#ADD8E6')  # Use the same light blue color as the line chart
        )

        # Create the highlight point chart
        highlight_chart = alt.Chart(highlight_point).mark_circle(size=100, color='red').encode(
            x='x:T',
            y='Price',
            tooltip=[alt.Tooltip('x_str', title='Date'), alt.Tooltip('Price', format='$,.0f')]  # Add tooltips for both x_str and y values
        )

        # Calculate percentiles
        min_price = chart_df['Price'].min()
        max_price = chart_df['Price'].max()
        range_40 = min_price + 0.40 * (max_price - min_price)
        range_70 = min_price + 0.70 * (max_price - min_price)

        if prediction <= range_40:
            desc = "For the selections you made prices can be lower. It is good time to make the bookings!"
        elif prediction <= range_70:
            desc = "For the selections you made prices are in typical range."
        else:
            desc = "Predicted price is higher. Analyze the trend or look deeper into insights to understand how different selections impact the price."
        

        st.markdown(f"""
        <div style='text-align: left;'>
        <h6 style='margin: 0; padding: 0;'> {desc} </h6>
        <h6 style='margin: 0; padding: 0;'> </h6>      
        <h6 style='margin: 0; padding: 0;'> </h6>
        </div>
        """, unsafe_allow_html=True)

        # Dashed line at 50% range
        green_line = alt.Chart(pd.DataFrame({'range_40': [range_40]})).mark_rule(color='yellow', strokeDash=[3,3]).encode(
            y='range_40:Q')
        
        # Dashed line at 50% range
        yellow_line = alt.Chart(pd.DataFrame({'range_70': [range_70]})).mark_rule(color='red', strokeDash=[3,3]).encode(
            y='range_70:Q')

        # Layer the base chart with the highlight point
        chart = (line_chart + point_chart + highlight_chart + green_line + yellow_line).properties(
            width=800,
            height=400
        )

        # Display the chart
        st.altair_chart(chart, use_container_width=True)