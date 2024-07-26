import pandas as pd
import sys
import pathlib
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import LinearRegression


def holiday():
    print('get all holidays in 2024')
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2024-01-01', end='2024-12-31').to_pydatetime()
    holiday_df = pd.DataFrame(holidays, columns=['Holiday'])
    return holiday_df


def get_first_carbon_value(arr):
    if len(arr) > 0:
        return arr[0]
    else:
        return None
    
def get_second_carbon_value(arr):
    if len(arr) > 1:
        return arr[1]
    else:
        return None


def run_processed_data(merged_path, processed_path):
    print('processing data')

    merged_df = pd.read_csv(merged_path)    
    
    merged_df = merged_df[~merged_df['price'].isnull()]

    merged_df = merged_df.drop_duplicates(keep='first')

    merged_df['Report_Run_Time'] = pd.to_datetime(merged_df['Report_Run_Time'])
    merged_df['Report_Run_Time'] = merged_df['Report_Run_Time'].dt.date

    merged_df.loc[merged_df['carrier'].str.contains('Separate tickets'), 'carrier'] = 'Third Party'

    merged_df['Airport_Route'] = merged_df['from_loc'] + " - " + merged_df['to_loc']

    merged_df.loc[merged_df.from_loc == 'LAX','From_City'] = 'Los Angeles'
    merged_df.loc[merged_df.from_loc == 'LGA','From_City'] = 'New York City'
    merged_df.loc[merged_df.from_loc == 'EWR','From_City'] = 'New York City'
    merged_df.loc[merged_df.from_loc == 'JFK','From_City'] = 'New York City'
    merged_df.loc[merged_df.to_loc == 'LAX','To_City'] = 'Los Angeles'
    merged_df.loc[merged_df.to_loc == 'LGA','To_City'] = 'New York City'
    merged_df.loc[merged_df.to_loc == 'EWR','To_City'] = 'New York City'
    merged_df.loc[merged_df.to_loc == 'JFK','To_City'] = 'New York City'

    merged_df['City_Route'] = merged_df['From_City'] + " - " + merged_df['To_City']

    merged_df['from_timestamp'] = pd.to_datetime(merged_df['from_timestamp']).dt.time
    merged_df['to_timestamp'] = pd.to_datetime(merged_df['to_timestamp']).dt.time

    merged_df['from_date'] = pd.to_datetime(merged_df['from_date'] + ' 2024', format='%b %d %Y')
    merged_df['to_date'] = pd.to_datetime(merged_df['to_date'] + ' 2024', format='%b %d %Y')

    merged_df['from_timestamp_1'] = pd.to_datetime(merged_df['from_date']) + pd.to_timedelta(merged_df['from_timestamp'].astype(str))
    merged_df['to_timestamp_1'] = pd.to_datetime(merged_df['to_date']) + pd.to_timedelta(merged_df['to_timestamp'].astype(str))
    
    merged_df['from_hour'] = merged_df['from_timestamp_1'].dt.round('15min').dt.strftime('%H:%M')
    merged_df['to_hour'] = merged_df['to_timestamp'].apply(lambda x:x.hour)

    merged_df['flight_duration'] = merged_df['to_timestamp_1'] - merged_df['from_timestamp_1']
    merged_df.loc[merged_df['flight_duration'].dt.total_seconds() < 0, 'to_timestamp_1'] += pd.to_timedelta('1 day')
    merged_df['flight_duration'] = merged_df['to_timestamp_1'] - merged_df['from_timestamp_1']
    
    merged_df[['Hours', 'Minutes']] = merged_df['details'].str.extract(r'Total duration (\d+) hr(?: (\d+) min)?')
    merged_df['Minutes'] = merged_df['Minutes'].fillna(0).astype(int)
    merged_df['Hours'] = pd.to_numeric(merged_df['Hours'])
    merged_df['Minutes'] = pd.to_numeric(merged_df['Minutes'])
    merged_df['Flight_Time'] = merged_df['Hours'] + round(merged_df['Minutes']/60,1)

    holiday_df = holiday()
    merged_df.loc[merged_df['from_date'].isin(holiday_df['Holiday']),'Holiday'] = 'Holiday'
    merged_df.loc[~merged_df['from_date'].isin(holiday_df['Holiday']),'Holiday'] = 'Not_Holiday'

    merged_df['Days_to_Fly'] = merged_df['from_date'] - pd.to_datetime(merged_df['Report_Run_Time'])
    merged_df['Days_to_Fly'] = merged_df['Days_to_Fly'].dt.days

    merged_df['flight_duration_value'] = round(merged_df.flight_duration.dt.seconds/3600,1)
    merged_df.loc[merged_df['City_Route']=='New York City - Los Angeles', 'flight_duration_value'] = merged_df['flight_duration_value'] + 3
    merged_df.loc[merged_df['City_Route']=='Los Angeles - New York City', 'flight_duration_value'] = merged_df['flight_duration_value'] - 3
    merged_df.loc[merged_df['Flight_Time'].isnull(),'Flight_Time'] = merged_df['flight_duration_value']
    merged_df['Fly_WeekDay'] = merged_df['from_timestamp_1'].dt.weekday + 1

    merged_df.loc[~merged_df['carbon_emission'].str.contains('emissions'),'carbon_emission'] = ""
    merged_df['carbon_array'] = merged_df.carbon_emission.str.split(".")
    merged_df['Carbon emissions estimate'] = merged_df['carbon_array'].apply(lambda x: get_first_carbon_value(x))
    merged_df['carbon_emission_1'] = merged_df['carbon_array'].apply(lambda x: get_second_carbon_value(x))
    merged_df['carbon_emission%'] = merged_df['carbon_emission_1'].fillna(merged_df['carbon_emission'])
    merged_df['Carbon emissions estimate'] = merged_df['Carbon emissions estimate'].str.replace('Carbon emissions estimate: ','')
    merged_df['carbon_emission%'] = merged_df['carbon_emission%'].str.replace(' emissions','')
    merged_df.loc[merged_df['Carbon emissions estimate'].str.contains('%'),'Carbon emissions estimate'] = ''
    merged_df['Carbon emissions estimate'] = merged_df['Carbon emissions estimate'].str.replace(' kilograms','')    
    merged_df['Carbon emissions estimate num'] = pd.to_numeric(merged_df['Carbon emissions estimate'], errors='coerce').astype('Int64')
    merged_df.loc[(merged_df['Carbon emissions estimate'].str.contains('1'))
       | (merged_df['Carbon emissions estimate'].str.contains('2'))
       | (merged_df['Carbon emissions estimate'].str.contains('3'))
       | (merged_df['Carbon emissions estimate'].str.contains('4'))
       | (merged_df['Carbon emissions estimate'].str.contains('5'))
       | (merged_df['Carbon emissions estimate'].str.contains('6'))
       | (merged_df['Carbon emissions estimate'].str.contains('7'))
       | (merged_df['Carbon emissions estimate'].str.contains('8'))
       | (merged_df['Carbon emissions estimate'].str.contains('9')),
       'Carbon emissions estimate'] = merged_df['Carbon emissions estimate num']
    merged_df['carbon_emission% num'] = merged_df['carbon_emission%'].str.replace(',','')
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace('%','')
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace(' ','')
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace('Avg','0')
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace('Average','0')
    merged_df.loc[merged_df['carbon_emission% num'].str.contains('-'), 'carbon_emission% num symbol'] = '-'
    merged_df.loc[merged_df['carbon_emission% num'].str.contains('\+'), 'carbon_emission% num symbol'] = '+'
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace('-','')
    merged_df['carbon_emission% num'] = merged_df['carbon_emission% num'].str.replace('+','')
    merged_df['carbon_emission% num'] = pd.to_numeric(merged_df['carbon_emission% num'], errors='coerce').astype('Int64')
    merged_df.loc[merged_df['carbon_emission% num symbol'] == '+', 'carbon_emission% num'] = merged_df['carbon_emission% num'] + 100
    merged_df.loc[merged_df['carbon_emission% num symbol'] == '-', 'carbon_emission% num'] = 100 - merged_df['carbon_emission% num']
    merged_df.loc[merged_df['carbon_emission% num'] == 0, 'carbon_emission% num'] = 100

    merged_df.loc[merged_df['overhead_bin'].str.contains("doesn't include overhead bin access"), 'overhead_bin'] = 'Additional charge for overhead bin'
    merged_df.loc[merged_df['overhead_bin'].str.startswith("$"), 'overhead_bin'] = 'No additional charge for overhead bin'

    merged_df.loc[merged_df.round_trip_duration == 0, 'Trip_Type'] = 'One Way'
    merged_df.loc[merged_df.round_trip_duration > 0, 'Trip_Type'] = 'Rounds Trip'

    merged_df = merged_df[['Report_Run_Time', 'carrier', 'Trip_Type', 
                           'Airport_Route','City_Route'
                        ,'from_timestamp_1', 'to_timestamp_1', 
                         'flight_duration_value', 'Fly_WeekDay',
                         'stop', 'price','overhead_bin', 'round_trip_duration',
       'Carbon emissions estimate num', 'carbon_emission% num','Days_to_Fly','Holiday']]
    
    merged_df = merged_df.drop_duplicates(keep='first')

    train_carbon_df = merged_df[~(merged_df['Carbon emissions estimate num'].isnull()) 
                         & ~(merged_df['carbon_emission% num'].isnull())][['Carbon emissions estimate num','carbon_emission% num']]
    test_carbon_df = merged_df[(merged_df['Carbon emissions estimate num'].isnull()) 
                               & ~(merged_df['carbon_emission% num'].isnull())][['Carbon emissions estimate num','carbon_emission% num']]
    test_carbon_df = test_carbon_df.reset_index()

    model = LinearRegression()
    model.fit(train_carbon_df[['carbon_emission% num']],train_carbon_df[['Carbon emissions estimate num']])
    carbon_y_pred = model.predict(test_carbon_df[['carbon_emission% num']])
    carbon_y_pred = pd.DataFrame(carbon_y_pred, columns=['Carbon emissions estimate num pred'])
    carbon_y_pred = round(carbon_y_pred)    
    carbon_df = pd.concat([test_carbon_df,carbon_y_pred], axis=1)
    carbon_df = carbon_df[['index', 'carbon_emission% num',
       'Carbon emissions estimate num pred']]
    merged_df = merged_df.reset_index()
    merged_df = pd.merge(merged_df, carbon_df[['index', 'Carbon emissions estimate num pred']], on= 'index', how='left')
    merged_df.loc[merged_df['Carbon emissions estimate num'].isnull(), 'Carbon emissions estimate num'] = merged_df['Carbon emissions estimate num pred']
    merged_df = merged_df[~merged_df['Carbon emissions estimate num'].isnull()]
    merged_df = merged_df[['Report_Run_Time', 'carrier', 'Trip_Type', 'Airport_Route',
   'price', 'overhead_bin',
      'round_trip_duration', 'Carbon emissions estimate num', 'carbon_emission% num',
         'Days_to_Fly', 'from_timestamp_1', 'to_timestamp_1',
         'flight_duration_value', 'Holiday', 'Fly_WeekDay', 'stop']]
    
    merged_df['from_hour'] = merged_df['from_timestamp_1'].dt.round('15min').dt.strftime('%H:%M')
    
    merged_df = merged_df[[ 'carrier', 'Trip_Type','Airport_Route',
          'stop','round_trip_duration','Days_to_Fly',
     'from_hour', 'flight_duration_value',
        'Holiday', 'Fly_WeekDay', 'price']]
    merged_df = merged_df.drop_duplicates(keep='first')
    merged_df.to_csv(processed_path)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    processed_path = home_dir.as_posix() + sys.argv[2]
    processed_test_train_path = home_dir.as_posix() + r'/data/processed'
    merged_path = home_dir.as_posix() + sys.argv[1]
    run_processed_data(merged_path, processed_path) 

if __name__ == "__main__":
    main()