import pandas as pd
import numpy as np
import math
import time
import os
import example
from statistics import get_statistics
def save_results(d,save_path):
    d = pd.Series(d)
    df = d.to_frame().T
    with open(save_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

def get_dict(model_str,seed,constant_regularization,kreg):
    # change to string
    if kreg == 0:
        kreg_str = False
    else:
        kreg_str = True

    if constant_regularization == True:
        type_str = "RAPS"
    else:
        type_str = "QRAPS"

    # create dict with the parameters
    d = {"model": model_str,
         "seed": seed,
         "type": type_str,
         "kreg": kreg_str}

    return d

def check_trial_exist(df_origin,d):
    df_tmp = df_origin[(df_origin.model == d["model"]) & (df_origin.seed == d["seed"]) & (df_origin.type == d["type"]) & (
                df_origin.kreg == d["kreg"])]
    if df_tmp.empty:
        skip_flag = False
    else:
        skip_flag = True
    return skip_flag
def run_main_wrapper(out_dir, num_trials = 100 , constant_regularization_list=[True , False] , kreg_list=[None, 0] , model_str_list=["resnet152"],continue_run_path=None):
    if not continue_run_path:
        # create new result dir and save path
        out_dir = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir,exist_ok=True)
        save_path = os.path.join(out_dir, "results.csv")
    else:
        save_path = continue_run_path
        df_origin = pd.read_csv(save_path)

    # loop on the parameters
    seed_list = np.arange(num_trials)
    for model_str in model_str_list:
        for seed in seed_list:
            for kreg in kreg_list:
                for constant_regularization in constant_regularization_list:
                    d = get_dict(model_str, seed, constant_regularization, kreg)

                    if continue_run_path:
                        skip_flag = check_trial_exist(df_origin,d)
                        if skip_flag:
                            continue
                    print(d)
                    try:
                        top1_avg, top5_avg, coverage_avg, size_avg = example.main(seed = seed, kreg = kreg, constant_regularization = constant_regularization,model_str=model_str)
                    except:
                        top1_avg, top5_avg, coverage_avg, size_avg = math.nan , math.nan , math.nan ,math.nan

                    # add results to dict
                    d["top1_avg"] = top1_avg
                    d["top5_avg"] = top5_avg
                    d["coverage_avg"] = coverage_avg
                    d["size_avg"] = size_avg

                    # save result in realtime
                    save_results(d=d, save_path=save_path)
                    get_statistics(results_path = save_path, overide_bool=False)

    print("finish main wrapper")


if __name__ == "__main__":
    out_dir = "/MLdata/ml_roie_adi/results"
    num_trials = 50
    constant_regularization_list = [True, False]
    kreg_list = [None, 0]
    # model_str_list = ["resnet152", "resnet18", "inception_v3", "shufflenetv2", "resnet50", "vgg16" , "densenet161"]
    model_str_list = ["resnet152"]
    continue_run_path = "/MLdata/ml_roie_adi/results/20220725-094057/results.csv"
    run_main_wrapper(out_dir=out_dir, num_trials = num_trials , constant_regularization_list=constant_regularization_list , kreg_list=kreg_list , model_str_list=model_str_list,continue_run_path=continue_run_path)