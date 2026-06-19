import pandas as pd

def text_to_string(path : str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        countries = file.read().replace('\n', '').split(',')
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

def add_form_team_features(df : pd.DataFrame, n : int = 10) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    home = pd.DataFrame({
        "match_id" : df.index, 
        "date" : df["date"],
        "team" : df["home_team"],
        "is_home" : True,
        "goals_for" : df["home_score"],
        "goals_against" : df["away_score"],

        "points" : 
        3 * (df["home_score"] > df["away_score"]).astype(int) +
        1 * (df["home_score"] == df["away_score"])
    })

    away = pd.DataFrame({
        "match_id" : df.index, 
        "date" : df["date"],
        "team" : df["home_team"],
        "is_home" : True,
        "goals_for" : df["home_score"],
        "goals_against" : df["away_score"],

        "points" : 
        3 * (df["home_score"] < df["away_score"]).astype(int) +
        1 * (df["home_score"] == df["away_score"])
    })

    long = pd.concat(
        [home, away],
        ignore_index=True
    )

    long = long.sort_values(
        ["team", "date"]
    )

    grouped = long.groupby("team")

    long["avg_points_last_n"] = (
        grouped["points"]
        .transform(
            lambda x:
            x.shift()
            .rolling(
                n, 
                min_periods=1
            ).mean()
        )
    ) 

    long["avg_goals_last_n"] = (
        grouped["goals_for"]
        .transform(
            lambda x:
            x.shift()
            .rolling(
                n,
                min_periods=1
            ).mean()
        )
    )

    long["avg_conceded_last_n"] = (
        grouped["goals_against"]
        .transform(
            lambda x:
            x.shift()
            .rolling(
                n,
                min_periods=1
            ).mean()
        )
    )

def filter_data_func(max_date : str = None) -> None:

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
    
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df["year"] = results_df["date"].dt.year

    if max_date != None:
        results_df = results_df[results_df["date"] < max_date]
        
    return results_df
    