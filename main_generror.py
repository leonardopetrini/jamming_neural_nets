"""
Version of the main file to study generalization error
"""

from experiment import *

L = 7
P = int(2 ** 14)

for r_min, N in zip([3.1], [5320]): #,2,2.5]: # eventually choose the rs s.t. respect ns in models

    parameters = {'P': P,
                  'datatype': 'ov',
                  'networktype': 'rect',
                  'r_min': r_min,
                  'L': L,
                  'new_data': False
                  }

    exp1 = Experiment(**parameters)

    exp1.dataset.load(0)
    exp1.dataset.build_dataset(N, exp1.main_dir)

    # exp1.find_transition(r_min,
    #                      P=0,
    #                      L=0,
    #                      layer_to_train=0,
    #                      jamming_margin=2,
    #                      gen_error_flag=False)

    exp1.save()

    for l in range(1, len(exp1.dataset)-1):

        exp1.find_transition(r_min=0.5,
                             P=0,
                             L=3,
                             layer_to_train=l,
                             jamming_margin=2,
                             gen_error_flag=True)
        exp1.save()

        try:
            exp1.find_transition(r_min=1,
                                 P=0,
                                 L=L-l,
                                 layer_to_train=l,
                                 jamming_margin=2,
                                 gen_error_flag=False)
        except TypeError:
            print('>>>>>>>>> Too easy dataset <<<<<<<<<!!!')
        exp1.save()
