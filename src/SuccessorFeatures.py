import numpy as np
from src.BaseModel import BaseModel

class SuccessorFeatures(BaseModel):
    """
    Successor Features model

    Arguments
    ---------
    env : object
        Environment object
    id : int
        Agent ID
    model_label : str
        Label for the model. Set to whatever name you want to identify
        the model.
    alpha : float
        Learning rate, bounded [0, 1]
    beta : float
        Inverse temperature parameter in the softmax function. A higher
        values produces more deterministic choice.
    gamma : float
        Discount parameter. Higher "looks" further into the future
    lmbd : float
        Decay parameter. Higher values decay feature predictions
        to 0.
    lmbd_l1 : float
        L1 regularization parameter. Higher values enforce sparse
        representation.
    bias_magnitude : float
        Magnitude of bias on successor matrix learning 
    bias_accuracy : float
        How accurate semantic bias matrix is to category overlap.
        Bounded [0, 1]
    inference_power : float
        Degree to which the successor matrix is reweighted according
        to a power function during value function computation.
    conjunctive_starts : bool
        If True, use discrete one-hot encoding of start states.
        If False, use feature-based encoding of start states.
    conjunctive_successors : bool
        If True, use discrete one-hot encoding of successor states.
        If False, use feature-based encoding of successor states.
    conjunctive_composition : bool
        If True, analyze conjunctions of feature options across
        feature categories during composition.
        If False, choose between each feature category independently.
    memory_sampler: bool
        If False, only retrieve exact matches in memory during inference
        If True, sample memories during inference based on similarity,
        recency, and frequency.
    sampler_feature_weight: float
        Weight of feature similarity in sampling, bounded [0, 1]
    sampler_recency_weight: float
        Weight of state update recency in sampling, bounded [0, 1]
    sampler_specificity : float
        Degree to which sampling is biased towards the most similar
        matches in memory, bounded [1, inf)
    """

    def __init__(
        self,
        env = None,
        id = 0,
        model_label = 'SuccessorFeatures',
        alpha = 1.,
        beta = np.inf,
        gamma = 1.,
        lmbd = 0.,
        lmbd_l1 = 0.,
        bias_magnitude = 0,
        bias_accuracy = 1.,
        inference_power = 1.0,
        conjunctive_starts = False,
        conjunctive_successors = False,
        conjunctive_composition = False,
        memory_sampler = False,
        sampler_feature_weight = .5,
        sampler_recency_weight = .5,
        sampler_specificity = 1.
    ):
        
        # Set model name and label
        self.model = 'SuccessorFeatures'
        self.model_label = model_label
        
        # Initialize base model
        super().__init__(
            env,
            id,
            alpha,
            beta,
            gamma,
            lmbd,
            lmbd_l1,
            bias_magnitude,
            bias_accuracy,
            inference_power,
            conjunctive_starts,
            conjunctive_successors,
            conjunctive_composition,
            memory_sampler,
            sampler_feature_weight,
            sampler_recency_weight,
            sampler_specificity
        )

    def compute_V(self):
        """
        Computed estimated value function based on successor matrix, M
        and current task, w
        """

        # N observations yet
        if len(self.S) == 0:
            self.V = []
            return
        
        # Use the successor matrix directly
        if self.inference_power == 1:
            M = self.M

        # Compute value function within power-based reweighting of the
        # rows of the successor matrix
        else:
            M_rowsum = np.sum(np.abs(self.M), axis=1)
            M_inference = np.sign(self.M)*(self.M**self.inference_power)
            sum_denom = np.sum(M_inference, axis=1, keepdims=True)
            zero_rows = (sum_denom == 0).flatten()
            sum_denom[zero_rows] = 1
            M_inference[zero_rows] = self.M[zero_rows]
            M_inference = M_inference/sum_denom
            M_inference = M_rowsum.reshape(-1, 1)*M_inference
            M = M_inference

        # Compute value function
        self.V = M@self.w


    def get_feature_vector(self, state):
        """
        Get feature vector for successor matrix update

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array
        
        Returns
        -------
        features : numpy.Array
            Feature vector for state
        """
        if self.conjunctive_starts == self.conjunctive_successors:
            features = np.eye(len(self.M))
        elif self.conjunctive_successors:
            features = self.get_state_index(state)
        elif not self.continuous_features:
            features = self.get_discrete_feature_index(state)
        else:
            features = state
        return features

    def update_M(self, state, state_new):
        """
        Update successor matrix (M)

        Arguments
        ---------
        state : numpy.Array
            One-dimensional current state array
        state : numpy.Array
            One-dimensional successor state array
        """

        # Get weights for rows of M for the present and successor states
        s_weight, s_new_weight = self.get_M_update_weights(state, state_new)

        # Prevent discounting on absorbing (terminal) transitions
        terminal = np.all(state == state_new)
        if terminal:
            bias = np.zeros_like(self.bias)
            gamma = 0.0
        else:
            bias = self.bias
            gamma = self.gamma

            # Weight and normalize bias based on the successor weight
            bias = self.weight_bias_matrix(s_new_weight, bias)

        # Get feature representation in M for present state 
        features = self.get_feature_vector(state)

        # Perform update
        delta = features + gamma*bias@self.M - self.M
        self.M = (1 - self.lmbd*self.alpha*s_weight)*self.M
        self.M += self.alpha*s_weight*delta

        # Apply L1 regularization with soft-thresholding to enforce
        # sparse representation
        threshold = self.alpha*s_weight*self.lmbd_l1
        M_thresh = np.abs(self.M) - threshold
        M_zeros = np.zeros_like(self.M)
        self.M = np.sign(self.M)*np.max([M_thresh, M_zeros], axis=0)
