import pandas as pd
from statistics import get_statistics

def apply_union(paths):
    df = pd.DataFrame()

    for path in paths:
        df_tmp = get_statistics(results_path=path,overide_bool=False)
        df = pd.concat([df, df_tmp], axis=0)

    df.to_csv("/tcmldrive/adi/ml/results/stat_ver0.csv")

    df.drop(columns=["Top-1","Top-5","Count"],inplace=True)
    df.to_csv("/tcmldrive/adi/ml/results/stat_ver1.csv")
    pass

if __name__ == "__main__":

    # set paths to be list of all the results csv paths
    paths=[
        '/tcmldrive/adi/ml/results/20220725-094057/results.csv',
        '/tcmldrive/adi/ml/results/20220726-220953/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221018/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221152/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221243/results.csv',
        '/tcmldrive/adi/ml/results/20220726-221309/results.csv']
    apply_union(paths)
