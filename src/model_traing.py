import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler,LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger = logging.getLogger('model training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
log_file_path=os.path.join(log_dir,"model training.log")
file_handler=logging.FileHandler(log_file_path)

formatter =logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def model_training(x_train:pd.DataFrame,y_train:pd.DataFrame,y_test:pd.DataFrame,x_test:pd.DataFrame )->pd.DataFrame:
    try: 
         
        model = Sequential([
            Dense(128, input_dim=x_train .shape[1], activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')  # Multi-class classification
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        logger.debug('model summary written')
        history = model.fit(x_train, y_train,
                    validation_data=(x_test , y_test),
                    epochs=20, batch_size=32)
        logger.debug('model training done')
        return model,history

    except Exception as e:
        logger.error('unexpected error happened:%s',e)
        raise

def predictions(model, x_test):
    try:
        preds = model.predict(x_test)
        predicted_labels = np.argmax(preds, axis=1)
        logger.debug("Model prediction done")
        return predicted_labels

    except Exception as e:
        logger.error('Unexpected error happened: %s', e)
        raise

def main():
    try:
        test_df = pd.read_csv(r"D:\github projects\spotify-songs-recmmender-system-\data\interim\preprocess_test_data.csv")
        train_df = pd.read_csv(r"D:\github projects\spotify-songs-recmmender-system-\data\interim\preprocess_train_data.csv")

        # Separate features and target
        x_train = train_df.drop(columns=['track_name', 'track_name_encoded'])
        y_train = train_df['track_name_encoded']

        x_test = test_df.drop(columns=['track_name', 'track_name_encoded'])
        y_test = test_df['track_name_encoded']

        x_train = x_train.apply(pd.to_numeric, errors='coerce')
        x_test = x_test.apply(pd.to_numeric, errors='coerce')

        # Convert labels if they are not numeric
        y_train = pd.to_numeric(y_train, errors='coerce')
        y_test = pd.to_numeric(y_test, errors='coerce')

        model,history=model_training(x_train, y_train, y_test, x_test)
        pred=predictions(model, x_test)

    except Exception as e:
        logger.error('unexpected error happened:%s',e)
        raise
if __name__=='__main__':
    main()





