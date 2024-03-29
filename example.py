import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
from model_selections import select_model
import torch.backends.cudnn as cudnn
import random

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('--data', metavar='IMAGENETVALDIR', help='path to Imagenet Val',default="/tcmldrive/adi/data/imagenet/imagenet_val/")
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
# parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)
# optimize for 'size' or 'adaptiveness'
lamda_criterion = 'size'

def main(seed=0,kreg = None,constant_regularization = True,lamda=None,model_str="resnet152"):
    args = parser.parse_args()
    ### Fix randomness
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Get the conformal calibration dataset
    imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(
        torchvision.datasets.ImageFolder(args.data, transform), [args.num_calib, 50000 - args.num_calib])

    # Initialize loaders
    calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=args.batch_size, shuffle=True,
                                             pin_memory=True)

    cudnn.benchmark = True

    # Get the model
    # define gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0")

    # choose a model
    model = select_model(model_str=model_str)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    # if None, model will pick lamda on its own
    # lamda = None
    # define
    # kreg = None
    # use the normal RAPS regularization
    # constant_regularization = True
    # optimize for 'size' or 'adaptiveness'
    # lamda_criterion = 'size'
    # allow sets of size zero
    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True

    # Conformalize model
    model = ConformalModel(model, calib_loader, alpha=0.1, lamda=lamda, constant_regularization=constant_regularization,
                           randomized=randomized, allow_zero_sets=allow_zero_sets, kreg=kreg)

    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    top1_avg, top5_avg, coverage_avg, size_avg = validate(val_loader, model, print_bool=True)

    print("Complete!")
    return top1_avg, top5_avg, coverage_avg, size_avg
if __name__ == "__main__":
    lamda = None
    # define
    kreg = None
    # use the normal RAPS regularization
    constant_regularization = True
    model_str = "resnet18"
    main(kreg = kreg, constant_regularization = constant_regularization,lamda=lamda)
