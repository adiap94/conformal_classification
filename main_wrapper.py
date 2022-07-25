import numpy as np
import example


def run_main_wrapper(num_trials = 100 , constant_regularization_list=[True , False] , kreg_list=[None, 0] , model_str_list=["resnet152"]):
    seed_list = np.arange(num_trials)
    for model_str in model_str_list:
        for seed in seed_list:
            for kreg in kreg_list:
                for constant_regularization in constant_regularization_list:
                    example.main(seed = seed, kreg = kreg, constant_regularization = constant_regularization,model_str=model_str)
    print("finish main wrapper")


if __name__ == "__main__":

    num_trials = 50
    constant_regularization_list = [True, False]
    kreg_list = [None, 0]
    model_str_list = ["resnet152", "resnet18", "resnet50", "densenet161", "vgg16", "inception_v3", "shufflenetv2"]

    run_main_wrapper(num_trials = num_trials , constant_regularization_list=constant_regularization_list , kreg_list=kreg_list , model_str_list=model_str_list)