from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import random
import numpy as np
import pandas as pd

def add_index_label(df, target_column="target", index_columns="ind_num"):
    """
    Add an index label to the DataFrame based on the 'target' column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str, optional): The name of the target column to use for labeling (default is "target").
        index_columns (str, optional): The name of the column to store the index labels (default is "ind_num").

    Returns:
        pandas.DataFrame: The input DataFrame with the 'target' column converted to binary values
        and the 'ind_num' column populated with index labels.

    This function takes a DataFrame and adds an index label based on the 'target' column.
    The 'target' column is converted into binary values (0 or 1), and based on its values,
    the 'ind' column is encoded and stored in the 'ind_num' column.

    Example:
    >>> import pandas as pd
    >>> data = {'label': [0, 1, 0, 1, 0], 'ind': ['A', 'B', 'A', 'B', 'C']}
    >>> df = pd.DataFrame(data)
    >>> df = add_index_label(df)
    >>> print(df)
       label ind  target ind_num
    0      0   A       0       0
    1      1   B       1       1
    2      0   A       0       0
    3      1   B       1       1
    4      0   C       0       2
    """
    
    # Convert the 'label' column into binary values (0 or 1)
    df[target_column] = LabelBinarizer().fit_transform(df.label)
    
    # For rows where the 'target' column is 0, encode the 'ind' column and store the result in 'ind_num'
    df.loc[df[target_column] == 0, index_columns] = LabelEncoder().fit_transform(df.loc[df[target_column] == 0, "ind"])
    
    # For rows where the 'target' column is 1, encode the 'ind' column and store the result in 'ind_num'
    df.loc[df[target_column] == 1, index_columns] = LabelEncoder().fit_transform(df.loc[df[target_column] == 1, "ind"])
    
    # Return the DataFrame with the updated columns
    return df
from typing import Tuple, Union

def train_test_split(df: pd.DataFrame, test_size: float = 0.33, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into training and testing sets based on a specified test size and an audio index.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float, optional): The proportion of the data to include in the test split. Default is 0.33.
        seed (int, optional): Seed for reproducibility. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.

    This function performs a stratified split, ensuring that the proportion of fake (target=0) and real (target=1) samples
    in both the training and testing sets closely matches the proportion in the original DataFrame.

    Example:
    train, test = train_test_split(df, test_size=0.2, seed=123)
    """
    num_fake_audios = len(set(df[df.target == 0].ind_num))
    num_real_audios = len(set(df[df.target == 1].ind_num))
    random.seed(seed)
    
    # Sample indices for fake and real audios
    num_train_fake_audios = int(np.ceil(num_fake_audios * (1 - test_size)))
    num_train_real_audios = int(np.ceil(num_real_audios * (1 - test_size)))

    fake_audios_index = random.sample(range(0, num_fake_audios), num_train_fake_audios)
    real_audios_index = random.sample(range(0, num_real_audios), num_train_real_audios)
    
    # Create the training set
    train = df[
        ((df.target == 0) & (df.ind_num.isin(fake_audios_index))) 
        | 
        ((df.target == 1) & (df.ind_num.isin(real_audios_index)))
    ]
    
    # Create the testing set
    test = df[
        ((df.target == 0) & (~df.ind_num.isin(fake_audios_index))) 
        | 
        ((df.target == 1) & (~df.ind_num.isin(real_audios_index)))
    ]
    
    return train, test
