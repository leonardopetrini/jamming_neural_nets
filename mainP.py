"""
Version of the main file for huge P to obtain the plot N vs P
"""
from experiment import *

L = 4
r_min = 2
P = int(2**19)

parameters = {'P': P,
              'datatype': 'ov',
              'networktype': 'recta',
              'r_min': r_min,
              'L': L,
              'new_data': True
              }

exp1 = Experiment(**parameters)

#  >>>>> Remember to break in for loop <<<<<

exp1.find_transition(r_min,
                    P=0,
                    L=0,
                    layer_to_train=0,
                    jamming_margin=2,
                    gen_error_flag=False)

exp1.save()

r_min = 1.8

P_list = np.logspace(np.log10(2**13), np.log10(P), 10, dtype=np.int32)

for p in P_list:
    exp1.find_transition(r_min,
                         P=p,
                         L=2,
                         layer_to_train=3,
                         jamming_margin=2,
                         gen_error_flag=False)
    exp1.save()



