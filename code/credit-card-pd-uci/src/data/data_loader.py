import os
import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # UCI Credit Risk dataset loading logic goes here
        # 1. read from /dataset/credit_risk.csv
        if os.path.exists(self.data_path):            
            df = pd.read_csv(self.data_path)
        else:
            # 2. if not exist, use uci package to download and load the dataset and save it to /dataset/credit_risk.csv
            df = self.download_and_save_dataset()

        # 3. return the loaded dataset as a pandas DataFrame
        return df
    
    def download_and_save_dataset(self):
        # Download the dataset using the uci package
        from ucimlrepo import fetch_ucirepo 
  
        # fetch dataset 
        default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
        # data (as pandas dataframes) 
        X = default_of_credit_card_clients.data.features 
        y = default_of_credit_card_clients.data.targets 
  
        # metadata 
        print(default_of_credit_card_clients.metadata) 
        print(default_of_credit_card_clients.variables) 

        # Save the dataset to /dataset/credit_risk.csv using stable column names
        # ensure the directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        # ucimlrepo returns feature matrices (X) but the variable metadata may
        # contain rich tuples/objects. Use stable X1..X23 names for columns so
        # downstream code (DataCleaner) sees the expected names.
        col_names = [f"X{i}" for i in range(1, 24)]
        df = pd.DataFrame(X, columns=col_names)
        # DataCleaner expects the target column to be named 'y'
        df['y'] = y
        # Overwrite any existing file (fixes previously-saved malformed CSV)
        df.to_csv(self.data_path, index=False)
        return df