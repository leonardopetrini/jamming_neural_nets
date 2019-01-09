"""
Version of the main file to use triangular networks
"""

from experiment import *

L = 3
r_min = 5
P = int(2 ** 13)  # int(2**20)

parameters = {'P': P,
              'datatype': 'ov',
              'networktype': 'trian',
              'r_min': r_min,
              'L': L,
              }
exp1 = Experiment(**parameters)

exp1.find_transition(r_min,
                     P=0,
                     L=0,
                     layer_to_train=0,
                     jamming_margin=2,
                     gen_error_flag=False)

exp1.save()

r_min = 2.8

for l in range(1, len(exp1.dataset)-1):

    exp1.find_transition(r_min=0.5,
                         P=0,
                         L=0,
                         layer_to_train=l,
                         jamming_margin=2,
                         gen_error_flag=True)
    exp.save()
    exp1.find_transition(r_min=2.2,
                         P=0,
                         L=0,
                         layer_to_train=l,
                         jamming_margin=2,
                         gen_error_flag=False)
    exp1.save()
