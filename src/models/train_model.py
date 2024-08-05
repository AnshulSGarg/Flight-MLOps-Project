import pandas as pd 
import numpy as np
import pathlib
import os
import yaml
import sys
from dvclive import Live
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

columns_to_encode = ['carrier','Trip_Type','Airport_Route','stop','Holiday','from_hour','Fly_WeekDay']
columns_to_scale = ['round_trip_duration', 'Days_to_Fly', 'flight_duration_value']

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
raw_path = home_dir.as_posix() 
pickle_path = raw_path  + r'/pickle_files/'
params_file = home_dir / 'params.yaml'
params = yaml.safe_load(open(params_file))["make_dataset"]
processed_path = home_dir.as_posix() + sys.argv[1]
processed_test_train_path = home_dir.as_posix() + r'/data/processed'


def split_data(df, split, seed):
    print('run split data')
    train, test = train_test_split(df, split, seed)
    return train, test

def save_data(train, test, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


def main():
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
        ('regressor', RandomForestRegressor(n_estimators=25))
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
    with open(pickle_path + 'flight_df.pkl', 'wb') as f:   
    # Dump the data into the pickle file
        pickle.dump(processed_df, f)

    print(r'saving flight_pipeline pickle file')
    with open(pickle_path + 'flight_pipeline.pkl', 'wb') as f:
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