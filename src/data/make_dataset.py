# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
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
# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()


import pandas as pd
import numpy as np
import pathlib
import yaml
import sys
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from dvclive import Live
import pickle
import warnings

# Ignore specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DataConversionWarning)

def split_data(df, split, seed):
    print('run split data')
    train, test = train_test_split(df, split, seed)
    return train, test

def save_data(train, test, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

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

def run_merge_csv_files(folder_path,merged_path):
    print('merging all raw data')
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.reset_index()
    merged_df = merged_df[['Report_Run_Time', 'carrier', 'from_loc',
       'to_loc', 'stop', 'price', 'from_timestamp', 'to_timestamp',
       'from_date', 'to_date', 'carbon_emission', 'overhead_bin', 'layover',
       'details', 'round_trip_duration']]
    merged_df.to_csv(merged_path)
    
def run_processed_data(merged_path, processed_path):
    print('processing data')

    merged_df = pd.read_csv(merged_path)    
    
    # drop columns with price value missing
    merged_df = merged_df[~merged_df['price'].isnull()]

    # drop duplicate columns from source_df
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
    # merged_df['to_hour'] = merged_df['to_timestamp'].apply(lambda x:x.hour)
    # merged_df.loc[(merged_df['to_hour'] >= 0) & (merged_df['to_hour'] < 3),'to_hour_segment'] = '1. 12 AM - 3 AM'
    # merged_df.loc[(merged_df['to_hour'] >= 3) & (merged_df['to_hour'] < 6),'to_hour_segment'] = '2. 3 AM - 6 AM'
    # merged_df.loc[(merged_df['to_hour'] >= 6) & (merged_df['to_hour'] < 9),'to_hour_segment'] = '3. 6 AM - 9 AM'
    # merged_df.loc[(merged_df['to_hour'] >= 9) & (merged_df['to_hour'] < 12),'to_hour_segment'] = '4. 9 AM - 12 PM'
    # merged_df.loc[(merged_df['to_hour'] >= 12) & (merged_df['to_hour'] < 15),'to_hour_segment'] = '5. 12 PM - 3 PM'
    # merged_df.loc[(merged_df['to_hour'] >= 15) & (merged_df['to_hour'] < 18),'to_hour_segment'] = '6. 3 PM - 6 PM'
    # merged_df.loc[(merged_df['to_hour'] >= 18) & (merged_df['to_hour'] < 21),'to_hour_segment'] = '7. 6 PM - 9 PM'
    # merged_df.loc[(merged_df['to_hour'] >= 21) & (merged_df['to_hour'] < 24),'to_hour_segment'] = '8. 9 PM - 12 AM'

    merged_df['from_date'] = pd.to_datetime(merged_df['from_date'] + ' 2024', format='%b %d %Y')
    merged_df['to_date'] = pd.to_datetime(merged_df['to_date'] + ' 2024', format='%b %d %Y')

    merged_df['from_timestamp_1'] = pd.to_datetime(merged_df['from_date']) + pd.to_timedelta(merged_df['from_timestamp'].astype(str))
    merged_df['to_timestamp_1'] = pd.to_datetime(merged_df['to_date']) + pd.to_timedelta(merged_df['to_timestamp'].astype(str))

    # merged_df['from_hour'] = merged_df['from_timestamp'].apply(lambda x:x.hour)
    # merged_df['from_timestamp_1'] = pd.to_datetime(merged_df['from_timestamp_1'])
    merged_df['from_hour'] = merged_df['from_timestamp_1'].dt.round('15min').dt.strftime('%H:%M')
    # merged_df['from_date'] = pd.to_datetime(merged_df['from_date'] + ' 2024', format='%b %d %Y')
    merged_df['to_hour'] = merged_df['to_timestamp'].apply(lambda x:x.hour)
    # merged_df['to_date'] = pd.to_datetime(merged_df['to_date'] + ' 2024', format='%b %d %Y')

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

    # merged_df['flight_duration'] = merged_df['to_timestamp_1'] - merged_df['from_timestamp_1']
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

    # merged_df.loc[merged_df['layover'].str.contains("(1 of 3)", na=False), 'stop'] = 3
    # merged_df.loc[merged_df['layover'].str.contains("(1 of 2)", na=False), 'stop'] = 2
    # merged_df.loc[merged_df['layover'].str.contains("(1 of 1)", na=False), 'stop'] = 1
    # merged_df['stop'] = merged_df['stop'].fillna(0)

    merged_df.loc[merged_df.round_trip_duration == 0, 'Trip_Type'] = 'One Way'
    merged_df.loc[merged_df.round_trip_duration > 0, 'Trip_Type'] = 'Rounds Trip'

    merged_df = merged_df[['Report_Run_Time', 'carrier', 'Trip_Type', 
                           'Airport_Route','City_Route'
                        #    ,'stop', 'from_hour_segment', 'to_hour_segment'
                        ,'from_timestamp_1', 'to_timestamp_1', 
                         'flight_duration_value', 'Fly_WeekDay',
                         'stop', 'price','overhead_bin', 'round_trip_duration',
       'Carbon emissions estimate num', 'carbon_emission% num','Days_to_Fly','Holiday']]
    
    merged_df = merged_df.drop_duplicates(keep='first')

    # merged_df = merged_df[['Report_Run_Time', 'carrier', 'Trip_Type', 'Airport_Route',
    #      'price', 'overhead_bin','stop',
    #    'round_trip_duration', 'Carbon emissions estimate num',
    #    'carbon_emission% num', 'Days_to_Fly', 'from_timestamp_1',
    # #    'to_timestamp_1', 'from_hour_segment', 'to_hour_segment',
    #    'flight_duration_value','Holiday','Fly_WeekDay']]

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
   #  'stop', 'from_hour_segment', 'to_hour_segment',
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

columns_to_encode = ['carrier','Trip_Type','Airport_Route','stop','Holiday','from_hour','Fly_WeekDay']
columns_to_scale = ['round_trip_duration', 'Days_to_Fly', 'flight_duration_value']

# print(f'input_file is {input_file}')
# data_path = home_dir.as_posix()
# print(f'data_path id {data_path}')
# output_path = home_dir.as_posix()
# print(f'output_path is {output_path}')
# data = load_data(data_path)
# print(f'main data is {data.shape}')
# # train_data, test_data = train_test_split(data, test_size=params['test_split'], random_state=params['seed'])
# print(f'train_data is {train_data.shape}')




def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    params_file = home_dir / 'params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    raw_path = home_dir.as_posix() + r'/data/raw'
    merged_path = home_dir.as_posix() + sys.argv[1]
    
    processed_path = home_dir.as_posix() + sys.argv[2]
    processed_test_train_path = home_dir.as_posix() + r'/data/processed'

    run_merge_csv_files(raw_path,merged_path)
    run_processed_data(merged_path, processed_path) 
       
    processed_df = pd.read_csv(processed_path, index_col=False)


    processed_df = processed_df[(processed_df['Days_to_Fly']>1)]
    # df = df[(df['Days_to_Fly']<76)]
    processed_df = processed_df[(processed_df['flight_duration_value'] > 4.4)]
    processed_df = processed_df[(processed_df['flight_duration_value'] < 12)]
    processed_df = processed_df[~(processed_df['carrier'].isin(['Third Party', 'Frontier', 'Sun Country Airlines']))]
    processed_df = processed_df[processed_df['price']<1000]

    # y_train = np.ravel(y_train)

    # # # K-fold cross-validation
    # # kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    # # scores = cross_val_score(pipeline, X, Y, cv= kfold, scoring='r2')
    # # print(scores.mean(),scores.std())

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),columns_to_scale),
            # ('cat', OrdinalEncoder(), columns_to_encode)
            ('cat', OneHotEncoder(), columns_to_encode)
        ],
        remainder='passthrough'
)

    # Creating a pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])
    pipeline

    # X_train = train_data[['carrier', 'Trip_Type', 'Airport_Route', 'round_trip_duration', 'stop',
    # 'Days_to_Fly', 'from_hour', 'flight_duration_value', 'Holiday',
    # 'Fly_WeekDay']]    
    # y_train = train_data[['price']]

    X = processed_df[['carrier', 'Trip_Type', 'Airport_Route', 'stop',
       'round_trip_duration', 'Days_to_Fly', 'from_hour',
       'flight_duration_value', 'Holiday', 'Fly_WeekDay']]
    y = processed_df[['price']]

    train_data, test_data = train_test_split(processed_df,test_size=params['test_split'],random_state=params['seed'])
    save_data(train_data, test_data, processed_test_train_path)

    X_train = train_data[['carrier', 'Trip_Type', 'Airport_Route', 'stop',
       'round_trip_duration', 'Days_to_Fly', 'from_hour',
       'flight_duration_value', 'Holiday', 'Fly_WeekDay']]
    y_train = train_data[['price']]
    X_test = test_data[['carrier', 'Trip_Type', 'Airport_Route', 'stop',
       'round_trip_duration', 'Days_to_Fly', 'from_hour',
       'flight_duration_value', 'Holiday', 'Fly_WeekDay']]
    y_test = test_data[['price']]

    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state= 42)
    print('training model')
    pipeline.fit(X_train, np.log1p(y_train))
    y_pred = pipeline.predict(X_test)
    
    print('making prediction')
    mae = mean_absolute_error(y_test[['price']],np.expm1(y_pred))
    # mae = mean_absolute_error(y_test,y_pred)
    print(f'mae is {mae}')
    r2 = r2_score(y_test[['price']],np.expm1(y_pred))
    # r2 = r2_score(y_test,y_pred)
    print("R2:", round(r2,2))

    # # Live.init("logs")

    # param_grid = {
    # # 'regressor__n_estimators': [100, 300],
    # 'regressor__n_estimators': [100],
    # # 'regressor__max_depth': [None, 30],
    # # 'regressor__max_samples':[0.1, 0.5, 1.0],
    # # 'regressor__max_features': ['log2', 'sqrt', None]
    #  'regressor__max_features': ['log2', 'sqrt']
    # }

    # kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    # from sklearn.model_selection import RandomizedSearchCV

    # grid_search = RandomizedSearchCV(pipeline, param_grid, cv=kfold, scoring='r2', n_jobs=-1, verbose=4, n_iter=5)

    # grid_search.fit(X_train, y_train)
    # final_pipe = grid_search.best_estimator_

    with Live() as live:
        # live.log_params(param_grid)
        live.log_metric("mae", mae)
        live.log_metric("R2", r2)
    live.end()

    print(r'saving flight_df pickle file')
    with open(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\flight_df.pkl', 'wb') as f:   
    # Dump the data into the pickle file
        pickle.dump(processed_df, f)

    print(r'saving flight_pipeline pickle file')
    with open(r'C:\Users\anshu\Desktop\MLOps\Flight-MLOps-Project\Flight-MLOps-Project\pickle_files\flight_pipeline.pkl', 'wb') as f:
    # Dump the data into the pickle file
        pickle.dump(pipeline, f)



    # # Finish dvclive logging
    # Live.save("final")



    # # Initialize dvclive
    # with Live() as live:
    #     # Log parameter grid
    #     live.log_params(param_grid)

    #     # Track best score and parameters
    #     live.log_metric("best_score", grid_search.best_score_)
    #     # live.log_params("best_params", grid_search.best_params_)

    #     # Track decision tree depth
    #     tree_depths = [estimator.tree_.max_depth for estimator in final_pipe.named_steps['regressor'].estimators_]
    #     # live.log_histogram('tree_depths', tree_depths)

    #     hist, bin_edges = np.histogram(tree_depths, bins='auto')

    #     # # Log histogram data
    #     # live.log_metric("tree_depth_histogram", hist.tolist())

    #     # # Log histogram bin edges
    #     # live.log_metric("tree_depth_bin_edges", bin_edges.tolist())

    #     # Track number of estimators
    #     num_estimators = final_pipe.named_steps['regressor'].n_estimators
    #     live.log_metric('num_estimators', num_estimators)

    #     # Track feature importances
    #     feature_importances = final_pipe.named_steps['regressor'].feature_importances_
    #     live.log_plot('feature_importances', feature_importances)

    #     # Visualize individual decision trees
    #     from sklearn import tree
    #     import matplotlib.pyplot as plt

    #     # Choose an estimator from the final model
    #     chosen_tree = final_pipe.named_steps['regressor'].estimators_[0]

    #     # Visualize the tree
    #     plt.figure(figsize=(20, 10))
    #     tree.plot_tree(chosen_tree, filled=True)
    #     plt.savefig("decision_tree_visualization.png")
    #     plt.close()

    #     # Log the visualization file
    #     live.log_image("decision_tree_visualization", "decision_tree_visualization.png")

    # # Save the logs
    # Live.save("final")
    # # search.best_params_
    # # search.best_score_
    # # # final_pipe.fit(X,Y)

    # # with Live() as live:
    # #     live.log_params(param_grid)
    #     # live.log("best_score", grid_search.best_score_)
    #     # live.log("best_params", grid_search.best_params_)

    # # Finish dvclive logging
    # # Live.save("final")


if __name__ == "__main__":
    main()