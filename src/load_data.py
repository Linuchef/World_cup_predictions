import kagglehub
import pandas as pd
import shutil
from pathlib import Path

def load_data_func() -> None:

# Download latest version
    path = Path(kagglehub.dataset_download(
        "martj42/international-football-results-from-1872-to-2017"
        )
        )
    
    raw_dir = Path("data/raw")

    for file in path.iterdir():
        shutil.copy2(file, raw_dir)

    return 