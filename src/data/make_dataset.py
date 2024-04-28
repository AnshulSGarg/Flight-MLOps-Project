# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()


import pandas as pd
import pathlib
import yaml
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.tseries.holiday import USFederalHolidayCalendar
import os


def load_data(path):
    print('run load data')
    df = pd.read_csv(path)
    return df
def split_data(df, split, seed):
    print('run split data')
    train, test = train_test_split(df, split, seed)
    return train, test
def save_data(train, test, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

def merge_csv_files(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def holiday():
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


# Display the merged DataFrame

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    raw_path = home_dir.as_posix() + r'/data/raw'
    merged_path = home_dir.as_posix() + r'/data/interim'
    processed_path = home_dir.as_posix() + r'/data/processed'

    merged_df = merge_csv_files(raw_path)
    merged_df = merged_df.reset_index()
    merged_df = merged_df[['Report_Run_Time', 'carrier', 'from_loc',
       'to_loc', 'stop', 'price', 'from_timestamp', 'to_timestamp',
       'from_date', 'to_date', 'carbon_emission', 'overhead_bin', 'layover',
       'details', 'round_trip_duration']]
    merged_df.to_csv(merged_path + r'/merged_raw_data.csv')

    merged_df = merged_df[~merged_df['price'].isnull()]

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

    merged_df[['Hours', 'Minutes']] = merged_df['details'].str.extract(r'Total duration (\d+) hr(?: (\d+) min)?')
    merged_df['Minutes'] = merged_df['Minutes'].fillna(0).astype(int)
    merged_df['Hours'] = pd.to_numeric(merged_df['Hours'])
    merged_df['Minutes'] = pd.to_numeric(merged_df['Minutes'])
    merged_df['Flight_Time'] = merged_df['Hours'] + round(merged_df['Minutes']/60,1)

    merged_df['from_timestamp'] = pd.to_datetime(merged_df['from_timestamp']).dt.time
    merged_df['to_timestamp'] = pd.to_datetime(merged_df['to_timestamp']).dt.time

    merged_df['from_hour'] = merged_df['from_timestamp'].apply(lambda x:x.hour)
    merged_df['from_date'] = pd.to_datetime(merged_df['from_date'] + ' 2024', format='%b %d %Y')
    merged_df['to_hour'] = merged_df['to_timestamp'].apply(lambda x:x.hour)
    merged_df['to_date'] = pd.to_datetime(merged_df['to_date'] + ' 2024', format='%b %d %Y')

    holiday_df = holiday()
    merged_df.loc[merged_df['from_date'].isin(holiday_df['Holiday']),'Holiday'] = 'Holiday'
    merged_df.loc[~merged_df['from_date'].isin(holiday_df['Holiday']),'Holiday'] = 'Not_Holiday'
    merged_df['Days_to_Fly'] = merged_df['from_date'] - pd.to_datetime(merged_df['Report_Run_Time'])
    merged_df['Days_to_Fly'] = merged_df['Days_to_Fly'].dt.days

    merged_df['from_timestamp_1'] = pd.to_datetime(merged_df['from_date']) + pd.to_timedelta(merged_df['from_timestamp'].astype(str))
    merged_df['to_timestamp_1'] = pd.to_datetime(merged_df['to_date']) + pd.to_timedelta(merged_df['to_timestamp'].astype(str))

    merged_df['flight_duration'] = merged_df['to_timestamp_1'] - merged_df['from_timestamp_1']
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

    # merged_df.loc[merged_df['layover'].str.contains("(1 of 3)", na=False), 'layover_count'] = 3
    # merged_df.loc[merged_df['layover'].str.contains("(1 of 2)", na=False), 'layover_count'] = 2
    # merged_df.loc[merged_df['layover'].str.contains("(1 of 1)", na=False), 'layover_count'] = 1
    # merged_df['layover_count'] = merged_df['layover_count'].fillna(0)

    merged_df.loc[merged_df.round_trip_duration == 0, 'Trip_Type'] = 'One Way'
    merged_df.loc[merged_df.round_trip_duration > 0, 'Trip_Type'] = 'Rounds Trip'

    merged_df = merged_df[['Report_Run_Time', 'carrier', 'Trip_Type', 
                           'Airport_Route','City_Route'
                        #    ,'layover_count', 'from_hour_segment', 'to_hour_segment'
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
        #  'layover_count', 'from_hour_segment', 'to_hour_segment',
           'round_trip_duration', 'Carbon emissions estimate num', 'carbon_emission% num',
             'Days_to_Fly', 'from_timestamp_1', 'to_timestamp_1',
               'flight_duration_value', 'Holiday', 'Fly_WeekDay', 'stop']]
    

    merged_df['from_hour'] = merged_df['from_timestamp_1'].dt.round('15min').dt.strftime('%H:%M')
    
    merged_df = merged_df[[ 'carrier', 'Trip_Type','Airport_Route',
          'stop','round_trip_duration','Days_to_Fly',
     'from_hour', 'flight_duration_value',
        'Holiday', 'Fly_WeekDay', 'price']]

    merged_df = merged_df.drop_duplicates(keep='first')
    print(processed_path + r'/processed_raw_data.csv')
    merged_df.to_csv(processed_path + r'/processed_raw_data.csv')





    # params_file = home_dir / 'params.yaml'
    # params = yaml.safe_load(open(params_file))["make_dataset"]
    # print(f'params is {params}')

    # print(f'sys.argv is {sys.argv}')
    # input_file = sys.argv[1]
    # print(f'input_file is {input_file}')
    # data_path = home_dir.as_posix()
    # print(f'data_path id {data_path}')
    # output_path = home_dir.as_posix()
    # print(f'output_path is {output_path}')
    # data = load_data(data_path)
    # print(f'main data is {data.shape}')
    # train_data, test_data = train_test_split(data, test_size=params['test_split'], random_state=params['seed'])
    # print(f'train_data is {train_data.shape}')
    # save_data(train_data, test_data, output_path)
if __name__ == "__main__":
    main()