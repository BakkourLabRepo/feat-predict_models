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
    alpha_decay : float
        Degree to which learning rate decays based on state visitation
        frequency, bounded [0, inf)
    beta : float
        Inverse temperature parameter in the softmax function. A higher
        values produces more deterministic choice.
    gamma : float
        Discount parameter. Higher "looks" further into the future
    bias_magnitude : float
        Magnitude of bias on successor matrix learning 
    bias_accuracy : float
        How accurate semantic bias matrix is to category overlap.
        Bounded [0, 1]
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
        alpha_decay = 0,
        beta = np.inf,
        gamma = 1.,
        bias_magnitude = 0,
        bias_accuracy = 1.,
        inference_inhibition = 0,
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
            alpha_decay,
            beta,
            gamma,
            bias_magnitude,
            bias_accuracy,
            inference_inhibition,
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
        if len(self.S) == 0:
            self.V = []
        else:
            self.V = self.M@self.w

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

        # Bias matrix differs for terminal/absorbing states
        terminal = np.all(state == state_new)
        if terminal:
            bias = self.bias_terminal
        else:
            bias = self.bias

        # Get weights for rows of M for the present and successor states
        s_weight, s_new_weight = self.get_M_update_weights(state, state_new)

        # Weight and normalize bias matrix based on the successor weight
        bias = self.weight_bias_matrix(s_new_weight, bias)

        # Get feature representation in M for present state 
        features = self.get_feature_vector(state)

        # Decay learning rate by state/feature visitation frequency
        # Can account for inflated values in the successor matrix
        alpha = self.decay_alpha()

        # Perform update
        delta = features + self.gamma*bias@self.M - self.M
        self.M += alpha*s_weight*delta

