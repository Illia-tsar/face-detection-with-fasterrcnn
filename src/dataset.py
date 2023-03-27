import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_test(data_path, val_size):
    train_df = pd.read_csv(os.path.join(data_path, "train/train.csv"), delimiter=",")
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"), delimiter=",")

    test_names = test_df.Name.tolist()
    train_names, val_names = train_test_split(train_df.Name.tolist(), test_size=val_size, random_state=17)
    return train_names, val_names, test_names
