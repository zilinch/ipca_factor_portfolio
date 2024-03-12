import json
import pandas as pd
import numpy as np

def serialize_matrix(matrix):
    """
    Serializes a matrix into a JSON string.
    """
    if isinstance(matrix, np.ndarray):
        return json.dumps(matrix.tolist())
    elif isinstance(matrix, pd.DataFrame):
        return matrix.to_json()
    else:
        raise TypeError("Matrix must be a numpy array or a pandas DataFrame")
    
def deserialize_matrix_to_dataframe(json_str):
    """
    Deserializes a JSON string back into a pandas DataFrame.
    
    json_str:
        json_str (str): The JSON string representation of the DataFrame.
    
    Returns:
        pandas.DataFrame: The deserialized pandas DataFrame.
    """
    return pd.read_json(json_str)