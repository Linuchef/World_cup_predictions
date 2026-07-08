import pandas as pd
import numpy as np 

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
        "team" : df["away_team"],
        "is_home" : False,
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

    long["avg_points_last_10"] = (
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

    long["avg_goals_last_10"] = (
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

    long["avg_conceded_last_10"] = (
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

    home_features = (
        long[long["is_home"]]
        [
            [
            "match_id",
            "avg_points_last_10",
            "avg_goals_last_10",
            "avg_conceded_last_10"
            ]
        ]
    .rename(
        columns = {
            "avg_points_last_10" : 
            "home_avg_points_last_10",

            "avg_goals_last_10" :
            "home_avg_goals_last_10",

            "avg_conceded_last_10" :
            "home_avg_conceded_last_10"
        }
    )
    )

    away_features = (
        long[~long["is_home"]]
        [
            [
                "match_id",
                "avg_points_last_10",
                "avg_goals_last_10",
                "avg_conceded_last_10"
            ]
        ]
        .rename(
            columns={
                "avg_points_last_10":
                    "away_avg_points_last_10",

                "avg_goals_last_10":
                    "away_avg_goals_last_10",

                "avg_conceded_last_10":
                    "away_avg_conceded_last_10"
            }
        )
    )

    df = df.merge(
        home_features,
        left_index=True,
        right_on="match_id",
        how="left"
    )

    df = df.merge(
        away_features,
        on="match_id",
        how="left"
    )

    return df.drop(
        columns = ["match_id"]
    )

def filter_data_func(
        min_date : str = None,
        max_date : str = None
        ) -> None:

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
    if min_date != None: 
        results_df = results_df[results_df["date"] > min_date]

    results_df = add_form_team_features(results_df)
        
    return results_df
    
def tournament_k(tournament : str) -> float:
    tournament = tournament.lower()

    if "world cup" in tournament:
        return 60
    if "qualification" in tournament:
        return 40
    if "friendly" in tournament:
        return 20
    return 30

def add_elo_features(
        df : pd.DataFrame,
        initial_elo : float = 1500,
        home_advantage : float = 75
) -> pd.DataFrame:
    
    df = df.sort_values("date").copy()
    ratings = {}

    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_elo = ratings.get(home_team, initial_elo)
        away_elo = ratings.get(away_team, initial_elo)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        if row["neutral"]:
            home_adjusted_elo = home_elo
        else:
            home_adjusted_elo = home_elo + home_advantage

        expected_home = 1 / (1 + 10**((away_elo - home_adjusted_elo) / 400))

        if row["home_score"] > row["away_score"]:
            actual_home = 1.0

        elif row["home_score"] == row["away_score"]:
            actual_home = 0.5

        else:
            actual_home = 0.0 
        goals_diff = abs(row["home_score"] - row["away_score"])
        margin_multiplier = np.log(goals_diff + 1)

        k = tournament_k(row["tournament"])

        change = k * margin_multiplier * (actual_home - expected_home)

        ratings["home_team"] = home_elos + change
        ratings["away_team"] = away_elo - change

    df["home_elo"] = home_elos
    df["away_elos"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df