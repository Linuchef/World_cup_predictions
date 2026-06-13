import pandas as pd

def text_to_string(path : str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        countries = file.read().replace('\n', '').split(',')
    return countries
        

    return countries

def update_country_names (
        df : pd.DataFrame,
        df_former : pd.DataFrame,
        team_type: str,
        col_name : str
        ) -> pd.DataFrame:
    
    mapping = df_former.set_index("former")["current"]
    df[col_name] = (df[team_type].map(mapping).fillna(df[team_type]))
    
    return df


def filter_data_func() -> None:

    dir = "data/raw/"
    countries_list = text_to_string("countries_of_interest.txt")

    former_names_df = pd.read_csv(dir + 'former_names.csv')
    goalscorers_df = pd.read_csv(dir + 'goalscorers.csv')
    results_df = pd.read_csv(dir + 'results.csv')
    shootouts_df = pd.read_csv(dir + 'shootouts.csv')

    results_df = update_country_names(
        results_df, former_names_df, 
        "home_team", 
        "home_team")
    results_df = update_country_names(
        results_df, 
        former_names_df, 
        "away_team",
        "away_team")


    return results_df
    