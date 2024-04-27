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




# Display the merged DataFrame

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    raw_path = home_dir.as_posix() + r'/data/raw'
    merged_path = home_dir.as_posix() + r'/data/interim'
    print(raw_path)
    merged_df = merge_csv_files(raw_path)
    merged_df.to_csv(merged_path + r'/merged_raw_data.csv')


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