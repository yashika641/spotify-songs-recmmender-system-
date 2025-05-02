import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
log_dir ="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_preprocess")
logger.setLevel('DEBUG')

console_handler =logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path =os.path.join(log_dir,'data_preprocess.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# def load_data(data_url:str)->pd.DataFrame:
#     try:
#         df=pd.read_csv(data_url)
#         logger.debug('data loaded from %s',data_url)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('failed to parse csv file:%s',e)
#         raise
#     except Exception as e:
#         logger.error('unecpected error occured:%s',e)
#         raise

def preprocess(df:pd.DataFrame)->pd. DataFrame:
    try:
        scaler = MinMaxScaler()

        # Apply scaling
        numeric_columns = ['duration_ms', 'danceability', 'energy', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo', 'popularity']

        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
 
        
        le=LabelEncoder()
    
        label_cols = ['artists', 'album_name' , 'key' , 'track_genre','explicit', 'mode', 'time_signature']
        df['track_name_encoded'] = le.fit_transform(df['track_name'])
# Apply label encoding to each column
        for col in label_cols:
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure all values are strings
        return df

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

def main():
    try:
        train_data=pd.read_csv(r'D:\github projects\spotify-songs-recmmender-system-\data\raw\train.csv')
        test_data=pd.read_csv(r'D:\github projects\spotify-songs-recmmender-system-\data\raw\test.csv')
        logger.debug(' data loaded properly:')

        train_processed_data=preprocess(train_data)
        test_processed_data=preprocess(test_data)

        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'preprocess_train_data.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'preprocess_test_data.csv'),index=False)

        logger.debug('processed data sucessfully:%s',data_path)

    except FileNotFoundError as e:
        logger.error('file not found:%s',e)
        raise
    except Exception as e:
        logger.error('unexpected error:%s',e)
        raise
if __name__=='__main__':
    main()
