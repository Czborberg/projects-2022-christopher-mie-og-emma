def index_function(df):
    """ Function to change index column to years and deleting original index column

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 

    df = df.reset_index()
    df['Year'] = pd.DatetimeIndex(df['DATE']).year
    del df["DATE"]
    df = df.set_index('Year')

    return df