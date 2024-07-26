import pandas as pd
import pathlib
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    raw_path = home_dir.as_posix() + r'/data/raw'
    merged_path = home_dir.as_posix() + sys.argv[1]
    
    run_merge_csv_files(raw_path,merged_path)     
    
if __name__ == "__main__":
    main()