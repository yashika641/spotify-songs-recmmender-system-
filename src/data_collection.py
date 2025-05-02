import pandas as pd
import numpy as np
import os 
import logging
from sklearn.model_selection import train_test_split

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger = logging.getLogger('data_collection')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path=os.path.join(log_dir,"data_collection.log")
file_handler=logging.FileHandler(log_file_path)

formatter =logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(filename: str) -> pd.DataFrame:
    try:
        file_path = os.path.join("experiments", filename)
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred: %s", e)
        raise

# def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
#     try:
#         df.drop(columns= ['track_id', 'track_name', 'album_name', 'explicit', 'popularity', 'track_genre'],inplace=True)
#         logger.debug('data preprocess completed')
#         return df
#     except KeyError as e:
#         logger.error('missing column:%s',e)
#         raise
#     except Exception as e:
#         logger.error('unexpected error occured:%s',e)
#         raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('train and test data saved to:%s',raw_data_path)

    except Exception as e:
        logger.error('unexpected error ocurred:%s',e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = ("dataset.csv")  # Change to your local path
        df = load_data(data_path)  # Using the load_data function to load from experiments folder
        # final_df = preprocess_data(df)  # Assume you have preprocessing logic here
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data/')  # Save processed data
    except FileNotFoundError as e:
        logger.error('Data file not found at %s: %s', data_path, e)
        raise
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data ingestion part: %s', e)
        raise

if __name__=='__main__':
    main()
