from classes import *
from alice import *
from train import *


class Experiment:
    """Main class used to define an experiment with input parameters:

        - P, defining the number of samples in the dataset

        - datatype, > if 'ov' the first model is trained on D0 on the SAT phase (depending on r_min)
            i.e. we are in the over-parametrized phase. Data-sets extracted from activations are 'well' structured;
                    > if 'rc' we find the jamming for D0 and all the activation data-sets are extracted
                    in that condition and hence don't exhibit much structure.

        - networktype ['rectangular' or 'triangular'] defines the shape of the network employed.
            for specific info see Rectangular and Triangular classes.

        - r_min = P / N_max determines the maximum size (N_max) of the models employed.
            If datatype == 'ov' this r also defines the point in the SAT phase we are working at;
            if datatype == 'rc', N_max is reduced - i.e. the network size is reduced -
                until we reach N* jamming point.

        - L is defined only in case Rectangular networks are employed and defines their size.
            The number of layers in triangular nets automatically defined given the size, cfr. Triangular class.


        The experiment is saved in self.main_dir which specifies all relevant parameters.
                e.g.  for rectangular net, SAT phase, P = 2^13, L = 5, r_min = 1.0 we have

                    ../recNet_ovL05P8192r1.0/
                                            /figures/
                                                    /loss
                                            /models
                                            /data

        """

    def __init__(self, P, datatype, networktype, r_min,
                 L = 0, new_data=True):

        self.P = P
        self.L = L
        self.datatype = datatype

        if 'ri' in networktype:
            self.networktype = 'triangular'
        else:
            self.networktype = 'rectangular'

        self.r_min = r_min

        self.main_dir = f'{self.networktype[:3]}Net_{datatype}L{L:02}P{P:05}r{r_min:0.1f}'
        make_dir(self.main_dir)

        if self.networktype == 'triangular':
            d_max = find_h_triangular_net(P/r_min) + 1
        else:
            d_max = find_h_rectangular_net(0, self.L, self.P, self.r_min) + 1

        self.dataset = DatasetClass(P, d_max, self.main_dir)

        if new_data:
            self.dataset.new()

        self.results = ResultsClass(self.main_dir)

    def save(self):
        pickle_save(self.__dict__, 'experiment', self.main_dir)

    def load(self):
        self.__dict__ = pickle_load('experiment', self.main_dir)

    @timeit
    def find_transition(self, r_min: float,
                        P: int = 0,
                        L: int = 0,
                        layer_to_train: int = 0,
                        jamming_margin: int = 2,
                        gen_error_flag: bool = False) -> None:

        """Find jamming transition for specified parameters.

        If P is given and < self.P, the input is subsampled.
        If L is given and rectangular nets are employed, L hidden layers are used independently on
            the layer trained. Otherwise, if training a layer Dn, the network used is the residual from the
            original starting from layer n .
        layer_to_train = n defines Dn, the layer used in the training.
        jamming_margin is the number of steps into the UNSAT phase we do before declaring the jamming found.
        NOTE: taking steps into the UNSAT phase is very expensive in computational time!
        If gen_error_flag == True, the input is split into 80% - 20% to have a training and test set and gen_error is
        computed. Results are stored with ResultsClass, as for the rest (cfr. ResultsClass).
        """

        # For rectangular net define L
        if L == 0:
            L = self.L - layer_to_train

        print(f'\n\n>>> FINDING TRANSITION FOR D{layer_to_train} <<<')

        self.dataset.load(layer_to_train)

        results_label = f'D{layer_to_train}'

        if gen_error_flag:
            # If also gen_error is computed add the information to the results keys
            results_label += 'ge'

        # If P is varying from initial, keep track in results
        if P:
            # If a different P is employed, add information to the results keys
            results_label += f'P{P}'
            self.dataset.P = P
        else:
            P = self.P

        # Add a new entry to the results dictionary
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
            # FINDING TRANSITION...

            model.new()

            if d == 0:
                self.dataset.d = model.h

            r = P / model.N
            print(f'\n{self.networktype} Net - ', end="")
            model.print_structure()
            print(f'\n  - N = {model.N}, P = {P}, r = {r:0.4f}')

            # Train current model
            loss, N_delta, gen_error, model = train_model(model,
                                                          self.dataset,
                                                          learning_rate=model.lr(),
                                                          epochs_number=int(1e6),
                                                          gen_error_flag=gen_error_flag)

            # Always save running loss and N_delta
            plt.savefig(f'{self.main_dir}/figures/loss/L{L}N{model.N}P{self.P}.png', format='png')
            plt.close('all')

            # Continue increasing r until it finds transition. Take jamming_margin points after and stop
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
            results_dict['Ns_jamming'] = self.dataset.build_dataset(model.N_list[-jamming_margin-1])

