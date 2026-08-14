import numpy as np
from src.BaseModel import BaseModel

class MBRL(BaseModel):
    """
    Model-Based Reinforcement Learning agent

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
        model_label = 'MBRL',
        alpha = 1.,
        alpha_decay = 0,
        beta = np.inf,
        gamma = 1.,
        bias_magnitude = 0,
        bias_accuracy = 1.,
        conjunctive_starts = False,
        conjunctive_successors = False,
        conjunctive_composition = False,
        memory_sampler = False,
        sampler_feature_weight = .5,
        sampler_recency_weight = .5,
        sampler_specificity = 1.
    ):
        
        # Set model name and label
        self.model = 'MBRL'
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
            conjunctive_starts,
            conjunctive_successors,
            conjunctive_composition,
            memory_sampler,
            sampler_feature_weight,
            sampler_recency_weight,
            sampler_specificity
        )

    def compute_V(self, max_itr=1000, tol=1e-4):
        """
        Computed estimated value function with value iteration

        Arguments
        ---------
        max_itr : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        """

        # N observations yet
        if len(self.S) == 0:
            self.V = []
            return
               
        # Value iteration
        M_biased = self.bias*self.M
        self.V = np.zeros(len(self.M))
        for _ in range(max_itr):

            # Perform Bellman update
            V_new = self.w + self.gamma*M_biased@self.V

            # Check for convergence
            if np.max(np.abs(V_new - self.V)) < tol:
                break

            self.V = V_new

    def get_feature_vector(self, state):
        """
        Get feature vector for transition matrix update

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array
        
        Returns
        -------
        features : numpy.Array
            Feature vector for state
        """
        if self.conjunctive_successors:
            features = self.get_state_index(state)
        else:
            features = self.get_discrete_feature_index(state)
        features = features.astype(float)
        return features

    def update_M(self, state, state_new):
        """
        Update one-step transition matrix (M)

        Arguments
        ---------
        state : numpy.Array
            One-dimensional current state array
        state_new : numpy.Array
            One-dimensional successor state array
        """

        # Do not update on terminal transitions to improve stability
        # when gamma = 1 and the environment is episodic and
        # terminating
        terminal = np.all(state == state_new)
        if terminal:
            return

        # Get weights for rows of M for the present and successor states
        s_weight = self.get_M_update_weights(state, state_new)[0]

        # Get feature representation in M for next state 
        features_new = self.get_feature_vector(state_new)

        # Decay learning rate by state/feature visitation frequency
        # Can account for inflated values in the successor matrix
        alpha = self.decay_alpha()

        # Perform update
        delta = features_new - self.M
        self.M += alpha*s_weight*delta

