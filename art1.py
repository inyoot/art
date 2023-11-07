"""
Original ART system for binary inputs and adaptive resonance to identify matching output cluster.

This code follows the naming conventions found in book "Clustering" by Xu and Wunsch
"""

# References:
# [1] Xu and Wunsch "Clustering", IEEE Press, 2009, ISBN 978-0-470-27680-8, pp 116-120

import numpy as np
from numpy import ndarray

XI = 1.1  # ξ in Clustering book


class ART1:
    """A class implementing the ART1 binary classification approach"""
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
    random_seed: float  # random seed value for replicating tests
    shuffle: bool  # whether to shuffle input batch when training
    nbr_epochs: int  # number of epochs to train

    def __init__(self, nbr_inputs: int, vigilance: float):
        """
        Initialize parameters and collections
        :param nbr_inputs: size of input pattern f1
        :param vigilance: vigilance parameter for comparing to resonance to identify the matching cluster
        """
        super().__init__(nbr_inputs=nbr_inputs, vigilance=vigilance)
        '''
        F1 LAYER
        F1 is the feature vector of size nbr_inputs, i.e., the binary pattern of 0/1 values.
        Initialize to zero (arbitrary since since x is replaced at each run).
        Indexing: i
        Rather than explicitly create a F1 layer, the input pattern x is used 
        '''
        '''
        F2 LAYER
        F2 is the output vector of classification nodes.
        Initialize to one node whose is set to 1.
        Indexing: j
        This structure contains a flag indicating that it is in use (i.e., 'assigned').
        '''
        self.f2_assigned: ndarray = np.empty(1, dtype=int)  # use 0/1 values, 0 = False, 1 = True
        self.f2_assigned.fill(0)
        self.nbr_f2_assigned = self.f2_assigned.sum()
        '''
        WEIGHTS F1 -> F2
        w_12 is the 2-D matrix of weights from features in layer F1 to layer F2.
        Initialize each value to XI/(XI - 1 + nbr_inputs) (i.e., normalize).
        Also known as short-term memory (STM).
        Known as 'bottom-up' weights.
        Indexing: w[i, j] - row number i is F1 node position, column number j is F2 node position
        '''
        self.w_12: ndarray = np.empty((nbr_inputs, 1), dtype=float)  # create nbr_inputs F1 nodes, 1 F2 node
        init_weight: float = self._get_initial_weight_f1_to_f2()
        self.w_12.fill(init_weight)
        """
        WEIGHTS F2 -> F1
        w_21 is the 2-D matrix of weights from categories in layer F2 to the input features in layer F1.
        Also called long-term memory (LTM)
        Initialize to value of 1.
        Indexing: w[j, i] - row number j is F2 node position, column number i is F1 node position
        """
        self.w_21: ndarray = np.empty((1, nbr_inputs), dtype=float)  # create one F2 node, nbr_inputs F1 nodes
        self.w_21.fill(1.0)
        """
        Initialize x
        """
        self.x = np.zeros(self.nbr_inputs)

    def _get_initial_weight_f1_to_f2(self) -> float:
        """
        Get initial weight from F1 to F2
        :return: initial weight value
        """
        return XI / (XI - 1.0 + self.nbr_inputs)

    def _compute_t(self) -> int:
        # The @ operator performs element-wise multiplication
        self.t = self.x @ self.w_12  # T value in clustering book, equation 5.16.
        nbr_non_zero_values = len(np.where(self.t > 0.0)[0])
        return nbr_non_zero_values

    def _find_match(self,) -> tuple[bool, int]:
        """
        Find best match F2 node in existing nodes.
        Process the input features to identify the matching cluster in vector F2.
        If a match occurs with an existing F2 node, resonance occurs, i.e., result is greater than vigilance value.
        When resonance occurs, both bottom-up and top-down weights are updated.
        If resonance does not occur, then search to find better pattern in F2.
        :return:
        """
        '''
        Compute input from F1 to F2.
        Shown as T in clustering book.
        '''
        match_found = False
        winning_f2_pos: int = -1
        nbr_non_zero_values =self._compute_t()
        if nbr_non_zero_values == 0:
            return match_found, winning_f2_pos
        for _ in range(nbr_non_zero_values):  # iterate over all
            f2_node_pos = self.t.argmax()
            x_and_w_21 = self.x * self.w_21[f2_node_pos]  # multiplication for AND function since x is binary
            x_and_w_21_sum = x_and_w_21.sum()
            x_sum = self.x.sum()
            rho: float = x_and_w_21_sum / x_sum  # see Equation 5.18 in clustering book.
            if rho >= self.vigilance:
                match_found = True
                winning_f2_pos = f2_node_pos
                self.f2_assigned[winning_f2_pos] = 1
                self.nbr_f2_assigned = self.f2_assigned.sum()
                break
            else:
                self.t[f2_node_pos] = 0.0  # remove this node from consideration
        return match_found, winning_f2_pos

    def _initialize_wgt_new_f2_node(self):
        # add initialized weights to weight matrix from F2 to F1
        new_21_weights = np.empty((1, self.nbr_inputs))
        new_21_weights.fill(1.0)
        self.w_21 = np.append(self.w_21, new_21_weights, axis=0)
        # add initialized weights to weight matrix from F1 to F2
        new_12_weights = np.empty((self.nbr_inputs, 1))
        new_12_weights.fill(self._get_initial_weight_f1_to_f2())
        self.w_12 = np.append(self.w_12, new_12_weights, axis=1)


    def _update_existing_weights(self, j: int):
        """
        Update existing weights when previously assigned node j in F2 is chosen.
        Update weights from F1 to F2 and from F2 to F1.
        Uses Equations 5.19 and 5.20 from Clustering book
        :param j: index for neuron chosen in F2 layer, i.e., row j in weight matrix w_21
        :return:
        """
        '''
        Update weights from F2 to F1 for row j in w_21.
        Uses Equations 5.19 from Clustering book
        '''
        w_21_j = self.w_21[j]
        updated_w_21_j = self.x * w_21_j  # element-wise product as AND operation, equation 5.19 in Clustering book
        self.w_21[j] = updated_w_21_j  # replace old weights with new
        '''
        Update weights from F1 to F2 for column j in w_12.
        Uses equation 5.20 in Clustering book.
        '''
        updated_w_12_j = XI * updated_w_21_j/(XI - 1.0 + self.w_21.sum())  # Equation 5.20 in Clustering book
        self.w_12[:, j] = updated_w_12_j  # replace old weights with new

    def _add_new_f2_node(self) -> int:
        """
        Add new node to F2.
        :return: returns new node position
        """
        # assign new node to F2
        self.nbr_f2_assigned = self.f2.sum()
        # if this is first node assigned, use the first position
        if self.nbr_f2_assigned == 0:
            self.f2[0] = 1
            self.nbr_f2_assigned = 1
            self.is_first_f2_assignment = True
            self.selected_f2_idx = 0
            self.is_match_found = True
            return 0
        # if not first node assignment, then create new node and assignment
        self.is_first_f2_assignment = False
        self.selected_f2_idx = self.nbr_f2_assigned
        new_buffer = np.empty(1, dtype=int)
        new_buffer.fill(0)
        self.f2 = np.append(self.f2, new_buffer, axis=0)
        self.f2[self.selected_f2_idx] = 1
        self.nbr_f2_assigned = self.f2.sum()
        self.is_match_found = True

    def _update_weights(self, match_found: bool, winning_node: int) -> bool:
        if match_found:
            self._update_existing_weights(j=winning_node)
            is_new = False
        else:
            self._add_new_f2_node()  # assign additional node to F2 layer
            self._initialize_wgt_new_f2_node()
            is_new = True
        self.f2_assigned[winning_node] = 1
        return is_new

    def get_template(self, x: ndarray) -> tuple[int, bool]:
        """
        Find best matching node in F2, and compare to vigilance parameter to see if it meets resonance criteria.
        Process the input features to identify the matching cluster in vector F2.
        If a match occurs with an existing F2 node, resonance occurs, i.e., result is greater than vigilance value.
        When resonance occurs, both bottom-up and top-down weights are updated.
        If resonance does not occur, then search to find better pattern in F2.
        If no pattern found, then a new node (i.e., new cluster) is added to the output vector F2.
        Weight matrices are adjusted.
        :param x: new input pattern for F1 - an array of length nbr_inputs, index = i
        :return: tuple of position of winning cluster in vector F2 and boolean indicating if this is a new cluster
        """
        ''''
        Iterate over results to find best match
        '''
        self.x = x
        match_found, winning_node = self._find_match()
        is_new = self._update_weights(match_found=match_found, winning_node=winning_node)
        return winning_node, is_new

