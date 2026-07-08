import pandas as pd 
import numpy as np 
from src.train_model import make_data_ready_glm

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

def team_stats(matches : pd.DataFrame, team : str):

    goals_for = []
    goals_against = []
    points = []

    for _, row in matches.iterrows():
        if row["home_team"] == team:
            gf = row["home_score"]
            ga = row["away_score"]
        else:
            gf = row["away_score"]
            ga = row["home_score"]

        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            points.append(3)
        elif gf == ga:
            points.append(1)
        else:
            points.append(0)
    
    return {"avg_goals" : sum(goals_for) / len(matches), 
            "avg_conceded" : sum(goals_against) / len(matches), 
            "avg_points" : sum(points) / len(matches)}
        

def make_match_features(
        df : pd.DataFrame,
        home_team : str, 
        away_team : str,
        neutral : bool,
        tournament : str,
        year : int, 
        n : int = 10
        ) -> pd.DataFrame:
    
    df = df.sort_values("date")
    home_matches = df[
        (df["home_team"] == home_team) | 
        (df["away_team"] == home_team)].tail(n)
    
    away_matches = df[
        (df["home_team"] == away_team) |
        (df["away_team"] == away_team)
    ].tail(n)
    
    home_team_stats = team_stats(home_matches, home_team)
    away_team_stats = team_stats(away_matches, away_team)

    return pd.DataFrame([{
        "home_team" : home_team,
        "away_team" : away_team,
        "tournament" : tournament,
        "neutral" : neutral,
        "year" : year,
        "home_avg_points_last_10" : home_team_stats["avg_points"],
        "home_avg_goals_last_10" : home_team_stats["avg_goals"],
        "home_avg_conceded_last_10" : home_team_stats["avg_conceded"],
        "away_avg_points_last_10" : away_team_stats["avg_points"],
        "away_avg_goals_last_10" : away_team_stats["avg_goals"],
        "away_avg_conceded_last_10" : away_team_stats["avg_conceded"]
    }])
    

def predict_goals_scored(
        df : pd.DataFrame,
        home_model : any,
        away_model : any,
        home_team : str,
        away_team : str,
        tournament : str = "FIFA World Cup",
        neutral : str = True,
        year : int = 2026,
        ):
    
    predict_df = make_match_features(df, home_team, away_team, neutral, tournament, year)
    ready_data = make_data_ready_glm(predict_df)
    ready_data = ready_data.reindex(
        columns=home_model.feature_names_in_,
        fill_value=0
    )

    lambda_home = home_model.predict(ready_data)[0]
    lambda_away = away_model.predict(ready_data)[0]

    return lambda_home, lambda_away