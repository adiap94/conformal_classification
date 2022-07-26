import os
import time
import pandas as pd

def get_statistics(results_path,overide_bool=False):
    df = pd.read_csv(results_path)
    stat = df.groupby(["model", "type", "kreg"])["size_avg"].agg(mean='mean', median='median', num_trials="count")
    print(stat)

    if overide_bool:
        save_path = os.path.join(os.path.dirname(results_path),"statistics.csv")
    else:
        save_path = os.path.join(os.path.dirname(results_path),"statistics")
        os.makedirs(save_path,exist_ok=True)
        save_path = os.path.join(save_path,time.strftime("%Y%m%d-%H%M%S")+".csv")
    stat.to_csv(save_path)


if __name__ == "__main__":
    results_path = "/MLdata/ml_roie_adi/results/20220725-094057/results.csv"
    get_statistics(results_path=results_path)