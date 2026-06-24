from src.train_model import multiple_poisson, multiple_XGBoost, make_data_ready_xgb, make_data_ready_glm
from sklearn.metrics import mean_poisson_deviance
import pandas as pd

def test_models(df : pd.DataFrame) -> pd.DataFrame:

    feature_cols = [
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
        "away_avg_conceded_last_10",
    ]

    df = df.dropna(
        subset=feature_cols + ["home_score", "away_score"]
    )

    train_mask = df["year"] < 2025
    df_train = df[train_mask]
    df_test = df[~train_mask]

    models_XGB = multiple_XGBoost(df_train)
    models_GLM = multiple_poisson(df_train)

    x_test = df_test[feature_cols]
    x_test_glm = make_data_ready_glm(x_test)
    x_test_xgb = make_data_ready_xgb(x_test)

    x_test_glm = x_test_glm.reindex(
        columns=models_GLM[0].model.exog_names,
        fill_value=0
    )

    x_test_xgb = x_test_xgb.reindex(
        columns=models_XGB[0].feature_names_in_,
        fill_value=0
    )
    
    home_pred_GLM = models_GLM[0].predict(x_test_glm)
    away_pred_GLM = models_GLM[1].predict(x_test_glm)

    home_pred_XGB = models_XGB[0].predict(x_test_xgb)
    away_pred_XGB = models_XGB[1].predict(x_test_xgb)

    y_home_test = df_test["home_score"]
    y_away_test = df_test["away_score"]

    home_score_GLM = mean_poisson_deviance(
        y_home_test,
        home_pred_GLM
    )

    home_score_XGB = mean_poisson_deviance(
        y_home_test,
        home_pred_XGB
    )
    
    away_score_GLM = mean_poisson_deviance(
        y_away_test,
        away_pred_GLM
    )

    away_score_XGB = mean_poisson_deviance(
        y_away_test,
        away_pred_XGB
    )

    return pd.DataFrame({
        "glm_home_score" : [home_score_GLM],
        "xgb_home_score" : [home_score_XGB],
        "glm_away_score" : [away_score_GLM],
        "xgb_away_score" : [away_score_XGB]
    })