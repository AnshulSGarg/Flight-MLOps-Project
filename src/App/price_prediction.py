import pandas as pd
import streamlit as st
import pickle
import datetime
import numpy as np

def Price_Prediction():
    st.title("Price Prediction")
    st.header("Enter Input")

    with open(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\flight_df.pkl','rb') as file:
        df = pickle.load(file)

    with open(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\flight_pipeline.pkl','rb') as file:
        pipeline = pickle.load(file)

    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2024-01-01', end='2024-12-31').to_pydatetime()
    holiday_df = pd.DataFrame(holidays, columns=['Holiday'])

    col1, col2 = st.columns(2)
    with col1:
        carrier = st.selectbox('Carrier', df['carrier'].unique().tolist())
        from_date = st.selectbox('Start Date',[datetime.date.today() + datetime.timedelta(days=1) + datetime.timedelta(days=i) for i in range(90)])
        # Airport_Route = st.selectbox('Airport_Route', df['Airport_Route'].unique().tolist())
        From_Airport = st.selectbox('From Airport',['Los Angeles International Airport (LAX)','Newark International Airport (EWR)','John F Kennedy International Airport (JFK)','La Guardia Airport (LGA)'])
        if From_Airport == 'Los Angeles International Airport (LAX)':
            From_Airport_Code = 'LAX'
        elif From_Airport == 'Newark International Airport (EWR)':
            From_Airport_Code = 'EWR'
        elif From_Airport == 'John F Kennedy International Airport (JFK)':
            From_Airport_Code = 'JFK'
        elif From_Airport == 'La Guardia Airport (LGA)':
            From_Airport_Code = 'LGA'


        def float_to_time(hour_float):
            hour = int(hour_float)
            minute = int((hour_float - hour) * 60)
            return f"{hour:02d}:{minute:02d}"
        
        time_options = [float_to_time(i/4) for i in range(0, 96)]

        from_hour = st.selectbox('from_hour', time_options, index=27)

        # from_hour = st.selectbox('from_hour',[float(i/2) for i in range(0, 48)], index=12)
        
        # merged_df['from_hour'] = merged_df['from_timestamp_1'].dt.round('15min').dt.strftime('%H:%M')

        flight_duration_value = st.selectbox('Flight Duration (hour)',[float(i/2) for i in range(10, 25)])

    with col2:
        # Trip_Type = st.selectbox('Trip_Type', df['Trip_Type'].unique().tolist())
        Trip_Type = st.selectbox('Trip_Type', ['One Way','Rounds Trip'])
        if Trip_Type == 'Rounds Trip':
            to_date = st.selectbox('Return Date', [from_date + datetime.timedelta(days=1) + datetime.timedelta(days=i) for i in range(90)])
        else:
            to_date = st.selectbox('Return Date',[''])
        if from_date in holiday_df:
            Holiday = 'Holiday'
        else:
            Holiday = 'Not_Holiday'
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

        stop = st.selectbox('Stops', ['Nonstop','1 stop','2 stops'])
        # if layover_stop == 'Nonstop':
        #     layover_count = 0.0
        # elif layover_stop == '1 Stop':
        #     layover_count = 1.0
        # elif layover_stop == '2 Stop':
        #     layover_count = 2.0
        # elif layover_stop == '3 Stop':
        #     layover_count = 3.0

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

    if st.button('Predict Price'):
        prediction = pipeline.predict(one_df)
        prediction = np.expm1(prediction)
        lower_bound = round(prediction[0]) - 30
        upper_bound = round(prediction[0]) + 30
        st.text('Prediction price should be between ${} and ${}'.format(lower_bound,upper_bound))

