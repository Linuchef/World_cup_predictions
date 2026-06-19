import pandas as pd

def GAM_func(target : str, data : pd.DataFrame) -> any:
    target_df = pd.Series(data[target])

    
    return target_df

