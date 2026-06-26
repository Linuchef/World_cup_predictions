import pandas as pd 
from src.train_model import make_data_ready_glm

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
        if gf == ga:
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
        torunament : str,
        year : int, 
        n : int = 10
        ):
    
    df = df.sort_values("date")
    home_matches = df[
        df["home_team"] == home_team | 
        df["away_team"] == home_team].tail(n)
    
    away_matches = df[
        df["home_team"] == away_team |
        df["away_team"] == away_team
    ].tail(n)
    
    home_team_stats = team_stats(home_matches, home_team)
    away_team_stats = team_stats(away_matches, away_team)

    return pd.DataFrame({
        "home_team" : home_team,
        "away_team" : away_team,
        "tournament" : torunament,
        "neutral" : neutral,
        "year" : year,
        "home_avg_points_last_10" : home_team_stats["avg_points"],
        "home_avg_goals_last_10" : home_team_stats["avg_goals"],
        "home_avg_conceded_last_10" : home_team_stats["avg_conceded"],
        "away_avg_points_last_10" : away_team_stats["avg_points"],
        "away_avg_goals_last_10" : away_team_stats["avg_goals"],
        "away_avg_conceded_last_10" : away_team_stats["avg_conceded"]
    })
    

def predict_goals_scored(
        df : pd.DataFrame,
        model_type : any,
        home_team : str,
        away_team : str,
        tournament : str = "Fifa World Cup",
        neutral : str = True,
        year : int = 2026,
        ):
    
    predict_df = make_match_features(df, home_team, away_team, neutral, tournament, year)
    ready_data = make_data_ready_glm(predict_df)
    ready_data = ready_data.reindex(
        columns=model_type.model.exog_names,
        fill_value=0
    )

    lambda_home = model_type.predict(ready_data)[0]
    lambda_away = model_type.predict(ready_data)[1]

    return lambda_home, lambda_away