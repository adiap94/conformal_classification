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

    if bool_only_valid_seed:
        df = get_valid_seed_trials(df)
    #apply statisics
    stat = df.groupby(["model", "type", "kreg"])['top1_avg', 'top5_avg', "coverage_avg", "size_avg"].mean()
    count = df.groupby(["model", "type", "kreg"]).size()
    stat = pd.concat([stat, count], axis=1)

    mapping = {'top1_avg': "Top-1", 'top5_avg': "Top-5", 'coverage_avg': "Coverage", 'size_avg': "Size", 0: "Count"}
    stat = stat.rename(columns=mapping)
    # stat = df.groupby(["model", "type", "kreg"])["size_avg"].agg(mean='mean', median='median', num_trials="count")
    print(stat)

    if overide_bool:
        save_path = os.path.join(os.path.dirname(results_path),"statistics.csv")
    else:
        save_path = os.path.join(os.path.dirname(results_path),"statistics")
        os.makedirs(save_path,exist_ok=True)
        save_path = os.path.join(save_path,time.strftime("%Y%m%d-%H%M%S")+".csv")
    stat.to_csv(save_path)

    return save_path

if __name__ == "__main__":
    # results_path = "/tcmldrive/adi/ml/results/20220725-094057/results.csv"
    # get_statistics(results_path=results_path,overide_bool=False)

    paths=[
        '/tcmldrive/adi/ml/results/20220725-094057/results.csv',
        '/tcmldrive/adi/ml/results/20220726-220953/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221018/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221152/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221243/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221309/results.csv']

    stat_paths = [get_statistics(results_path=p,overide_bool=False) for p in paths]
    pass