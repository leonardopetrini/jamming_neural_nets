from classes import *


class Experiment:

    def __init__(self, P, datatype, networktype, r_min,
                 L = 0, new_data=True):

        self.P = P
        self.L = L
        self.datatype = datatype

        if 'ria' in networktype:
            self.networktype = 'Triangular'
        else:
            self.networktype = 'Rectangular'

        self.r_min = r_min

        self.main_dir = f'{networktype}Net_{datatype}L{L:02}P{P:05}r{r_min:0.1f}'
        make_dir(self.main_dir)

        if 'ria' in networktype:
            d_max = find_h_triangular_net(P/r_min) + 1
        else:
            d_max = find_h(0, self.L, self.P, self.r_min) + 1

        self.dataset = DatasetClass(P, d_max, self.main_dir)

        if new_data:
            self.dataset.new()

        self.results = ResultsClass(self.main_dir)

    def save(self):
        pickle_save(self.__dict__, 'experiment', self.main_dir)

    def load(self):
        self.__dict__ = pickle_load('experiment', self.main_dir)

    def find_transition(self,r_min,
                        P=0,
                        L=0,
                        layer_to_train=0,
                        jamming_margin=2,
                        gen_error_flag=False):

        # For rectangular net define L
        if L == 0:
            L = self.L - layer_to_train

        print(f'\n\n>>> FINDING TRANSITION FOR D{layer_to_train} <<<')

        self.dataset.load(layer_to_train)

        results_label = f'D{layer_to_train}'

        if gen_error_flag:
            results_label += 'ge'

        # If P is varying from initial, keep track in results
        if P:
            results_label += f'P{P}'
            self.dataset.P = P
        else:
            P = self.P
        results_dict = self.results.new_results(results_label)

        if layer_to_train == 0:
            d = 0
        else:
            d = self.dataset.D.shape[1]

        if 'ria' in self.networktype:
            model = Triangular(P, r_min, d)
        else:
            model = Rectangular(P, r_min, d, L)

        UNSATFlag = 0

        while UNSATFlag < jamming_margin:

            model.new()

            if d == 0:
                self.dataset.d = model.h

            r = P / model.N
            print(f'\n{self.networktype} Net - ', end="")
            model.print_structure()
            print(f'\n  - N = {model.N}, P = {P}, r = {r:0.4f}')

            loss, N_delta, model, gen_error = train_model(model,
                                                          self.dataset,
                                                          learning_rate=model.lr(),
                                                          epochs_number=int(1e6),
                                                          gen_error_flag=gen_error_flag)

            plt.savefig(f'{self.main_dir}/figures/loss/L{L}N{model.N}P{self.P}.png', format='png')
            plt.close('all')

            # Continue increasing r until it finds transition. Take 3 points after and stop
            if N_delta > 0.0021 * P:
                UNSATFlag += 1
            else:
                UNSATFlag = 0

            if not d:
                model.save(self.main_dir)

            if gen_error_flag:
                results_dict['gen_error'].append(gen_error)

            results_dict['loss'].append(loss)
            results_dict['N_delta'].append(N_delta)
            results_dict['N'].append(model.N)
            results_dict['r'].append(r)

            if (not d and self.datatype == 'ov') or gen_error_flag:
                # Dont need to find jamming, no jamming_margin needed
                jamming_margin = -1
                break

            model.reduce(steps=1)

        self.results.save()

        # If training the first layer construct dataset from activations
        if d == 0:
            self.dataset.build_dataset(model.N_list[-jamming_margin-1])
