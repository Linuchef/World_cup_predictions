import pandas as pd

def text_to_string(path : str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        countries = file.read().replace('\n', '').split(',')
    return countries
        

    return countries

def filter_on_countries(
        df : pd.DataFrame, 
        countries : list[str]
        ) -> pd.DataFrame:
    
    return

def filter_data_func() -> None:

    dir = "data/raw/"
    countries_list = text_to_string("countries_of_interest.txt")

    former_names_df = pd.read_csv(dir + 'former_names.csv')
    goalscorers_df = pd.read_csv(dir + 'goalscorers.csv')
    results_df = pd.read_csv(dir + 'results.csv')
    shootouts_df = pd.read_csv(dir + 'shootouts.csv')

    return results_df
    