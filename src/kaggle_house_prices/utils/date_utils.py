import pandas as pd

def get_date_from_columns(df, column_dictionary, default_values=None):
    """
    Constructs a datetime column from multiple columns present in `df`
    
    :param df: DataFrame containing the columns to be used to construct the datetime column (type: pd.DataFrame)
    :param column_dictionary: Dictionary containing the column names to be used to construct the datetime column (type: dict)
        - Keys: Datetime component names (type: str)
        - Values: Column names (type: str)
    :param default_values: Dictionary containing the default values to be used for the datetime components (type: dict, optional)
        - Keys: Datetime component names (type: str)
        - Values: Default values (type: int)
    :return: Datetime column (type: pd.Series)
    """
    default_columns = {"year": 2000, "month": 1, "day": 1}
    if default_values is not None:
        default_columns.update(default_values)
    
    columns = default_columns
    for k, v in column_dictionary.items():
        columns[k] = df[v]
    dates = pd.to_datetime(columns)
    
    return pd.Series(dates, index=df.index)