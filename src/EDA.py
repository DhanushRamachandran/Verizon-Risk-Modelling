import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadd import load_data
# quality analysis
def analyze_quality(data: pd.DataFrame, quality_column: str) -> None:
    """
    Analyze the quality of the dataset by plotting the distribution of a specified quality column.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    quality_column (str): The name of the column representing quality.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=quality_column, data=data)
    plt.title('Quality Distribution')
    plt.xlabel(f'{quality_column}')
    plt.ylabel('Count')
    plt.show()
# missing values analysis
def analyze_missing_values(data: pd.DataFrame) -> None:
    """
    Analyze missing values in the dataset by plotting a heatmap.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
# correlation analysis
def analyze_correlation(data: pd.DataFrame) -> None:        
    """
    Analyze the correlation between numerical features in the dataset by plotting a heatmap.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
    None
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
# distribution analysis
def analyze_distribution(data: pd.DataFrame, column: str) -> None:
    """
    Analyze the distribution of a specified numerical column by plotting a histogram.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    column (str): The name of the numerical column to analyze.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
# outlier analysis
def analyze_outliers(data: pd.DataFrame, column: str) -> None:  
    """
    Analyze outliers in a specified numerical column by plotting a boxplot.

    Parameters:
    data (pd.DataFrame): The input data as a pandas DataFrame.
    column (str): The name of the numerical column to analyze.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Outlier Analysis of {column}')
    plt.xlabel(column)
    plt.show()


if __name__=="__main__":
    # Example usage
    

    df = pd.read_csv("data/Verizon_Data.csv")
    df[["year","month","day"]] = df["app_date"].str.split("-", expand=True)
    df.drop(columns=["app_date"], inplace=True)
    df[["year","month","day"]] = df[["year","month","day"]].astype(int)
    df.columns
    df.rename(columns={"application_date":"app_date","fico":"cred_score","del90":"delinquency_90_days"},inplace=True)
    df.to_csv("data/Verizon_Data.csv",index=False)
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
        analyze_quality(df, quality_column=col) 
    # quality and overall analysis
    analyze_missing_values(df)
    analyze_correlation(df.select_dtypes(include=['float64', 'int64']))
    df["delinquency_90_days"] = df["delinquency_90_days"].map({"Performing/paid off":1,"90+ delinquent/default":0})
    analyze_distribution 
    # univariate analysis
    # money loss because of delinquency
    df["money_loss"] = df["device_cost"] - df["down_payment"]
    
