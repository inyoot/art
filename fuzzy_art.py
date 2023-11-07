"""
Fuzzy ART implementation

Uses complement coding of input nodes (F1) to avoid proliferation of output nodes (F2).
The input x is both normalized and complement-coded; this is not required prior to submitting a pattern x to code.
The normalization method is MinMaxScaler from sklearn.preprocessing.
"""
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import random
from src.utils.state import State
from src.utils.linear_algebra import fuzzy_and

""""
Alpha is the choice parameter used to break ties.  Note that as the vigilance parameter rho decrease, 
then alpha should decrease as well.
"""
DEFAULT_ALPHA = 0.8
DEFAULT_BETA = 0.1
DEFAULT_NBR_EPOCHS = 1000  # default number of training epochs
DEFAULT_RANDOM_SEED = 4.0


# References:
# [1] Xu and Wunsch "Clustering", IEEE Press, 2009, ISBN 978-0-470-27680-8
# [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast
#     stable learning and categorization of analog patterns by an adaptive
#     resonance system," Neural networks, vol. 4, no. 6, pp. 759-771, 1991.


class FuzzyArt:
    nbr_inputs: int  # number of nodes in F1 layer
    vigilance: float  # vigilance parameter for determining resonance
    alpha: float  # parameter for breaking ties between candidate prototype vectors in F2
    beta: float  # learning rate
    x: ndarray # array of feature data
    x_compl: ndarray  # input array
    w: ndarray  # weights for x values
    t: ndarray  # array containing the results when computing category choice
    f2: ndarray  # list of F2 nodes that have been assigned to current and prior x input patterns
    is_first_f2_assignment: bool  # indicates that this is the first assigned node in F2
    nbr_f2_assigned: int  # number of F2 nodes that have been assigned
    selected_f2_idx: int  # indicates which F2 node matches the x input pattern
    state: State  # process state for processing each F1 feature set
    random_seed: float  # random seed value for replicating tests
    shuffle: bool  # whether to shuffle input batch when training
    nbr_epochs: int  # number of epochs to train

    def __init__(self,
                 nbr_inputs: int,
                 vigilance: float,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA):
        """
        Instantiate Fuzzy ART class.
        :param nbr_inputs: size of input vector x
        :param vigilance: minimum resonance required to choose an F2 node
        :param alpha: tie-breaker for choosing best-matching F2 nodes given input x
        :param beta: learning rate
        """
        """
        In fuzzy ART, the two weight matrices in ART1 are subsumed into one weight matrix.
        To stay consistent with the Clustering book, the weight index i indicates the F2 node, and j the F1 node.

        Complement coding of the input is used to avoid category proliferation (see Clustering book [1] pp. 123-124,
        specifically equation 5.31, and the Carpenter paper [2] section 5.1).
        """
        super().__init__(nbr_inputs, vigilance)
        self.alpha = alpha
        self.beta = beta

        """
        F1 LAYER
        F1 is the feature vector of size nbr_inputs.  
        For fuzzy ART, these are continuous in the interval [0,1].
        Rather than explicitly create a F1 layer, the input pattern x is used 
        Note that x is complement coded, i.e., two column vector where the first column is x, and the second column
        is set to 1 - x
        Weights indexing: j

        F2 LAYER
        F2 is the output vector of classification nodes.
        Initialize to one node whose is set to 1.
        F2 complement is set to 1 - F2.
        Weights indexing: i
        This structure contains a flag indicating that it is in use (i.e., 'assigned').
        Using nomenclature from Clustering book [1], F2 is shown as y, with y complement in second column.
        """

        """
        Weight vector between F1 and F2
        As mentioned above, the two weight matrices in ART1 are subsumed into one weight matrix for Fuzzy ART.
        Indexing: w[i,j], where i is the F2 node and j is the F1 node <-- note different order from ART1
        For complement-coding, weight matrices are required corresponding to u and v columns in Clustering book [1], 
        see Eq. 5.31 on pp. 124.  Matrix w for weight matrices includes the complement in columns >= nbr inputs.
        """
        self.w = np.empty((1, self.nbr_inputs * 2), dtype=float)
        self.w.fill(1.0)
        self.w[:, range(nbr_inputs, nbr_inputs * 2)] = 0.0

    def _norm_and_complement_code_x(self) -> None:
        # validate dimensionality
        if self.x.shape[1] != self.nbr_inputs:
            raise ValueError(f'Number of features in x ({self.x.shape[1]}) is not equal to '
                             f'number of features specified when instantiating Fuzzy ART '
                             f'class (nbr_features = {self.nbr_inputs})')
        # normalize x
        x_norm = MinMaxScaler(feature_range=(0, 1)).fit_transform(self.x)
        self.x_compl = np.zeros((self.x.shape[0], 2* self.x.shape[1]), dtype=float)
        self.x_compl[:, 0:self.nbr_inputs] = x_norm[:, 0:self.nbr_inputs]
        # complement code x
        self.x_compl[:, self.nbr_inputs: 2 * self.nbr_inputs] = 1.0 - self.x_compl[:, 0:self.nbr_inputs]
        # update state
        self.state.is_x_complemented = True

    def _compute_t(self, f1_idx: int):
        t = np.zeros(self.w.shape[0], dtype=float)
        f1 = self.x_compl[f1_idx, :]
        for i in range(self.w.shape[0]):
            wgt_i = self.w[i, :]
            x_and_w = fuzzy_and(f1, wgt_i)
            x_and_w_sum = x_and_w.sum()
            w_sum = wgt_i.sum()
            t[i] = x_and_w_sum / (DEFAULT_ALPHA + w_sum)
        self.t = t

    def _find_match(self, f1_idx: int) -> tuple[bool, int]:
        if not self.state.can_select_existing_f2():
            raise ValueError("Inconsistent state, cannot select an existing F2 node")
        '''
        The following code mimics the same flow as in ART1 (see the _find_match function).
        The significant difference is the use of fuzzy operations and complement coding
        '''
        # find best matching F2 node
        self._compute_t(f1_idx=f1_idx)
        nbr_non_zero_values = len(np.where(self.t > 0.0)[0])
        f1 = self.x_compl[f1_idx, :]
        for _ in range(nbr_non_zero_values):
            f2_idx = self.t.argmax()  # f2_idx = the position of the node in the F2 layer with maximum resonance
            # compute rho - individual steps broken out for testing/debugging
            fuzzy_f1_and_wgt = fuzzy_and(f1, self.w[f2_idx, :])  # min of f1 and weight
            rho_numerator = fuzzy_f1_and_wgt.sum()
            rho_denominator = f1.sum()
            rho = rho_numerator / rho_denominator
            # compare rho to vigilance to determine if resonance has occurred
            if rho >= self.vigilance:
                return True, f2_idx
            else:
                self.t[f2_idx] = 0.0  # reset - remove this node from consideration
        #  no F2 node found that meets vigilance requirement
        return False, -np.inf

    def _get_initial_weight_f1_to_f2(self) -> float:
        # not used
        ...

    def _initialize_wgt_f2_zero_node(self):
        # per equation 5.33, initial weights are equal to complement-coded input x
        new_weights = self.x_compl[0, :]
        self.w[0] = new_weights
        self.state.is_wgt_updated = True

    def _initialize_wgt_new_f2_node(self):
        """
        Use fast-commit learning when adding a new node to F2, i.e., Equation 5.33
        :return:
        """
        # add new row of weights to self.w weight matrix; this is the fast-commit for new F2 nodes
        f1_idx = self.state.f1_idx
        # per equation 5.33, initial weights are equal to complement-coded input x
        x_values = self.x_compl[f1_idx, :]
        new_weights = np.zeros((1, self.nbr_inputs * 2), dtype=float)
        new_weights[0, :] = x_values
        self.w = np.concatenate((self.w, new_weights), axis=0)
        self.state.is_wgt_updated = True

    def _update_existing_weights(self):
        """
        Method for updating existing weights associated with a selected F2 node and based upon learning rate beta.
        This method is for the slow-recode portion of learning.
        :return:
        """
        # See Equation 5.25 in Clustering book
        # this is the slow-recode for existing F2 nodes
        if self.state.is_slow_recode_enabled and not self.state.can_update_existing_weights():
            raise ValueError('Inconsistent state: cannot update existing weights')
        selected_f2_idx = self.state.selected_f2_idx
        f1_idx = self.state.f1_idx
        weights_i_updated = (self.beta * (fuzzy_and(self.x_compl[f1_idx, :], self.w[selected_f2_idx, :])) +
                             (1.0 - self.beta) * self.w[selected_f2_idx, :])
        self.w[selected_f2_idx, :] = weights_i_updated
        self.state.is_wgt_updated = True

    def predict(self, x: ndarray[float]) -> dict:
        if x.shape[1] != self.nbr_inputs:
            raise ValueError(f'Length of input x is not equal to specified nbr of inputs = {self.nbr_inputs:,}')
        if x.shape[0] < 1:
            raise ValueError('Must have at least one feature set in data')
        self.state.reset()
        self.state.is_predict_only = True
        self.state.is_process_started = True
        self.x = x
        self._norm_and_complement_code_x()
        results = {}
        for f1_idx in range(x.shape[0]):
            is_match_found, selected_f2_idx = self._find_match(f1_idx=f1_idx)
            results[f1_idx] = {
                'is_match_found': is_match_found,
                'selected_f2_idx': selected_f2_idx
            }
            self.state.reset()
        return results

    def fit(self, x: ndarray,
            nbr_epochs: int = DEFAULT_NBR_EPOCHS,
            random_seed: float = DEFAULT_RANDOM_SEED):
        self.x = x
        self.nbr_epochs = nbr_epochs
        self.random_seed = random_seed
        self._norm_and_complement_code_x()
        '''
        This method uses logic similar to that in the Islam Elnabarawy implementation of Fuzzy ART 
        '''
        random.seed(self.random_seed)  # set random seed
        for _ in range(self.nbr_epochs):
            # randomize input
            indices = list(range(self.x_compl.shape[0]))
            random.shuffle(indices)
            is_first_obs = True
            for f1_idx in indices:
                self.state.reset()
                self.state.is_process_started = True
                self.state.f1_idx = f1_idx
                if is_first_obs:
                    self.state.set_f2_zero_node()
                    self._initialize_wgt_f2_zero_node()
                    is_first_obs = False
                    continue
                is_match_found, winning_f2_idx = self._find_match(f1_idx=f1_idx)
                if is_match_found:
                    self.state.set_selected_f2_node(idx=winning_f2_idx)
                    self._update_existing_weights()
                else:
                    self.state.add_and_select_new_f2_node()
                    self._initialize_wgt_new_f2_node()





