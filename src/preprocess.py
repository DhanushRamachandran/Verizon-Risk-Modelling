# preprocess the data for modeling
import pandas as pd
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data by handling missing values and encoding categorical variables.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    # Handle missing values by filling them with the mean of the column
    # convert date into year, month, day
    data[["year","month","day"]] = data["app_date"].str.split("-", expand=True) 
    data.drop(columns=["app_date","day"], inplace=True)

    X = data.drop(columns=["delinquency_90_days"])
    y = data["delinquency_90_days"]

    X=pd.get_dummies(X, drop_first=True)
    print("no of features after encoding:", X.shape[1])
    return X,y 


    
if __name__ == "__main__":
    data = pd.read_csv(r"data/Verizon_Data.csv")
    X,y = preprocess_data(data)
    X.to_csv("data/Preprocessed_Features.csv", index=False)

    
