import statsmodels.api as sm
import pandas as pd
import xgboost as xgb

def make_data_ready_glm(x : pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(
        x, columns=[
            "home_team",
            "away_team",
            "tournament"
        ],
        drop_first=True
    )

    x["neutral"] = x["neutral"].astype(int)
    x["year"] = x["year"].astype(int)
    
    x = sm.add_constant(x)
    x = x.astype(float)

    return x

def make_data_ready_xgb(x : pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(
        x, columns=[
            "home_team",
            "away_team",
            "tournament"
        ]
    )

    x["neutral"] = x["neutral"].astype(int)
    x["year"] = x["year"].astype(int)
    
    x = x.astype(float)

    return x

def poisson_model(y : pd.Series, x : pd.DataFrame) -> any:

    x = make_data_ready_glm(x)
    y = y.astype(float)

    model = sm.GLM(
        y,
        x, 
        family = sm.families.Poisson()
    ).fit()

    return model

def XGboost_model(y : pd.Series, x : pd.DataFrame) -> any:

    
    y = y.astype(float)
    x = make_data_ready_xgb(x)

    model = xgb.XGBRegressor(
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        learning_rate=0.01,
        max_depth=5,
        n_estimators=1000,
        colsample_bytree=0.8
    )

    model.fit(x, y)

    return model

def multiple_poisson(df : pd.DataFrame) -> list[any]:
    y1 = df["home_score"]
    y2 = df["away_score"]

    x = df[
        [
            "home_team", 
            "away_team", 
            "tournament", 
            "neutral", 
            "year",

            "home_avg_points_last_10",
            "home_avg_goals_last_10",
            "home_avg_conceded_last_10",

            "away_avg_points_last_10",
            "away_avg_goals_last_10",
            "away_avg_conceded_last_10"]
        ]

    home_score_mod = poisson_model(y1, x)
    away_score_mod = poisson_model(y2, x)

    return home_score_mod, away_score_mod

def multiple_XGBoost(df : pd.DataFrame) -> list[any]:
    y1 = df["home_score"]
    y2 = df["away_score"]

    x = df[
        [
            "home_team", 
            "away_team", 
            "tournament", 
            "neutral", 
            "year",

            "home_avg_points_last_10",
            "home_avg_goals_last_10",
            "home_avg_conceded_last_10",

            "away_avg_points_last_10",
            "away_avg_goals_last_10",
            "away_avg_conceded_last_10"]
        ]
    home_score_mod = XGboost_model(y1, x)
    away_score_mod = XGboost_model(y2, x)

    return home_score_mod, away_score_mod
