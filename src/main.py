from load_data import load_data_func
from filter_data import filter_data_func
from rate_calculations import GAM_func
import pandas as pd

df = filter_data_func()
print(GAM_func("home_team", df).head())
