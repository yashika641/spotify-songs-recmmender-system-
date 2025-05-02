import pandas as pd
import numpy as np
import os 
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

def feature_eng(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns='Unnamed: 0')
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(target.unique()), activation='softmax')  # Multi-class classification
        ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()