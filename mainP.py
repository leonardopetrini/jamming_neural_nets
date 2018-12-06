from experiment import Experiment

L = 3
r_min = 1.8
P = int(2**20)

parameters = {'P': P,
              'datatype': 'ov',
              'networktype': 'recta',
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

P_list = np.logspace(np.log10(2**13), np.log10(P), 10, dtype=np.int32)

for p in P_list:
    exp1.find_transition(r_min,
                         P=p,
                         L=0,
                         layer_to_train=1,
                         jamming_margin=2,
                         gen_error_flag=False)
    exp1.save()



