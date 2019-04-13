# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:48:15 2019

@author: SrivatsanPC
"""

import argparse
from args_config import args_check

parser = argparse.ArgumentParser()

parser.add_argument('--n_iter', action = 'store', default = 10000, type=int)
parser.add_argument('--n_context_max', action = 'store', default = 50, type = int)
parser.add_argument('--plot_after', action = 'store', default = 10000, type = int)
parser.add_argument('--loss_after', action = 'store', default = 100, type = int)
parser.add_argument('--h_size', action = 'store', default = 128, type = int)
parser.add_argument('--model_type', action = 'store', default = 'ANP')
parser.add_argument('--attention', action = 'store', default = 'multihead')
parser.add_argument('--kernel', action = 'store', default = 'SE')

#####BOOLEAN##########
parser.add_argument('--random_kernel_params', action='store_true', default = True, type = bool)
parser.add_argument('--SA_decoder', action='store_true', default = False, type = bool)
parser.add_argument('--SA_det_encoder', action='store_true', default = False, type = bool)
parser.add_argument('--SA_lat_encoder', action='store_true', default = False, type = bool)
parser.add_argument('--CA_lat_encoder', action='store_true', default = False, type = bool)

args  = parser.parse_args()
args_check(args)





