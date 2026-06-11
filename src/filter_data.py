import pandas as pd

def text_to_string(path : str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        countries = file.read().replace('\n', '').split(',')
        

    return countries

def filter_data_func(df : pd.DataFrame) -> None:
    return 
    