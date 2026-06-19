import statsmodels.api as sm
import pandas as pd

def poisson_model(y : pd.Series, x : pd.DataFrame) -> any:

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
    y = y.astype(float)
    x = x.astype(float)

    model = sm.GLM(
        y,
        x, 
        family = sm.families.Poisson()
    ).fit()

    return model

def multiple_poisson(df : pd.DataFrame) -> list[any]:
    y1 = df["home_score"]
    y2 = df["away_score"]

    x = df[["home_team", "away_team", "tournament", "neutral", "year"]]

    home_score_mod = poisson_model(y1, x)
    away_score_mod = poisson_model(y2, x)

    return home_score_mod, away_score_mod
