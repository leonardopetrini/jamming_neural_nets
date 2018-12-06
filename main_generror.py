from experiment import Experiment

L = 8
r_min = 1
P = int(2 ** 13)  # int(2**20)

parameters = {'P': P,
              'datatype': 'ov',
              'networktype': 'rect',
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

for l in range(1, len(exp1.dataset)-1):

    exp1.find_transition(r_min=1,
                         P=0,
                         L=4,
                         layer_to_train=l,
                         jamming_margin=2,
                         gen_error_flag=True)
    exp1.save()

    exp1.find_transition(r_min=2.5,
                         P=0,
                         L=4,
                         layer_to_train=l,
                         jamming_margin=2,
                         gen_error_flag=False)
    exp1.save()
