import os
import time
import pandas as pd

def get_valid_seed_trials(df):

    df=df.copy()
    df = df.dropna()
    seed_count_list = df.groupby(["seed"]).size()
    seed_valid_list = seed_count_list[seed_count_list == 4].index.to_list()
    df = df[df.seed.isin(seed_valid_list)]

    return df

def get_statistics(results_path,overide_bool=False,bool_only_valid_seed=True):
    df = pd.read_csv(results_path)
    df.replace(to_replace="inception_v3", value="inception\_v3", inplace=True)
    if bool_only_valid_seed:
        df = get_valid_seed_trials(df)
    mapping = {"model":"Model","type":"Type",'top1_avg': "Top-1", 'top5_avg': "Top-5", 'coverage_avg': "Coverage", 'size_avg': "Size"}
    df = df.rename(columns=mapping)

    #apply statisics
    stat = df.groupby(["Model", "Type", "kreg"])['Top-1', 'Top-5', "Coverage", "Size"].mean()
    count = df.groupby(["Model", "Type", "kreg"]).size()
    stat = pd.concat([stat, count], axis=1)
    mapping = {0: "Count"}
    stat = stat.rename(columns=mapping)

    stat_summary = stat.drop(columns=['Top-1', 'Top-5', 'Count'])
    stat_summary=stat_summary.unstack(level=1).unstack(level=1)
    stat_summary["Top-1"] = stat.iloc[0]["Top-1"]
    stat_summary["Top-5"] = stat.iloc[0]["Top-5"]
    stat_summary["Count"] = stat.iloc[0]["Count"]
    print(stat)

    if overide_bool:
        save_path = os.path.join(os.path.dirname(results_path),"statistics.csv")
    else:
        save_path = os.path.join(os.path.dirname(results_path),"statistics")
        os.makedirs(save_path,exist_ok=True)
        save_path = os.path.join(save_path,time.strftime("%Y%m%d-%H%M%S")+".csv")
    stat.to_csv(save_path)

    return stat_summary

if __name__ == "__main__":
    pass