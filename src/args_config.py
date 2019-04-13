# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:19:50 2019

@author: SrivatsanPC
"""

def args_check(args):
    if args.attention not in ['uniform','laplace','dot_product','multihead']:
        raise NameError('attention name invalid' )
    elif args.model_type not in ['ANP','NP']:
        raise NameError('model_type name invalid')
    elif args.kernel not in ['SE','PER','custom']:
        raise NameError('kernel name invalid')
    elif args.n_context_max <=3 :
        raise ValueError('Max number of context points should be greater than 3')
    
    print('Yay! All args are consistent')
