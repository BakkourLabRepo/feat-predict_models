import numpy as np

class SuccessorFeatures:
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
    beta_test : float
        Inverse temperature parameter for testing. This is a separate
        parameter from beta, so that beta can be fit to the training
        data, and beta_test can be used to evaluate the model on the
        test data.
    gamma : float
        Discount parameter. A higher is less future discounting ("looks"
        further into the future)
    segmentation : float
        Controls the degree of bias to learn within-features, bounded 
        [-1, 1]
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
        env,
        id = 0,
        model_label = 'Successor_Features',
        alpha = 1.,
        alpha_2 = 1.,
        alpha_decay = 0,
        beta = np.inf,
        beta_test = np.inf,
        gamma = 1.,
        segmentation = 0,
        segmentation_2 = 0,
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
        self.id = id
        self.model_label = model_label
        self.alpha = alpha
        self.alpha_2 = alpha_2
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.beta_test = beta_test
        self.gamma = gamma
        self.segmentation = segmentation
        self.segmentation_2 = segmentation_2
        self.bias_accuracy = bias_accuracy
        self.inference_inhibition = inference_inhibition
        self.conjunctive_starts = conjunctive_starts
        self.conjunctive_successors = conjunctive_successors
        self.conjunctive_composition = conjunctive_composition

        self.S = np.array([])
        self.F = np.array([])
        self.F_raw = np.array([])
        self.bias = np.array([[]])
        self.continuous_features = env.continuous_features
        self.n_insts = len(env.tmat)
        self.n_feats = env.n_feats
        self.n_per = env.n_per

         # Get sampler weights
        self.memory_sampler = memory_sampler
        self.sampler_specificity = sampler_specificity
        self.sampler_feature_weight = sampler_feature_weight
        self.sampler_recency_weight = sampler_recency_weight
        if memory_sampler:
            self.samp_weights = [

                # Feature similarity weight
                sampler_feature_weight, 

                # Recency weight
                (1 - sampler_feature_weight)*sampler_recency_weight,

                # Frequency weight
                (1 - sampler_feature_weight)*(1 - sampler_recency_weight)

            ]
        else:
            self.samp_weights = [None, None, None]

    def decompose_state(self, state):
        """
        Decompose 1d state array into 2d features array

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array
        
        Returns
        -------
        features : numpy.Array
            Two-dimensional features array, with one row per feature
            present in the input state
        """
        features = state*np.eye(len(state), dtype=int)
        features = features[state.astype(bool)]
        return features

    def binarize_state(self, state):
        """
        Convert state with discrete feature levels to binary encoding

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array
        
        Returns
        -------
        binarized : numpy.Array
            1-dimensional array, where each element corresponds to one
            feature level
        """
        binarized = np.zeros(self.n_insts*self.n_feats, dtype=int)
        for i, inst in enumerate(state):
            if inst != 0:
                binarized[i*self.n_insts + inst - 1] = 1
        return binarized

    def add_row_to_M(self):
        """
        Add row to successor matrix (M) and bias matrix (bias)
        """
        self.M = np.vstack((self.M, np.zeros(np.shape(self.M)[1])))

    def add_col_to_M(self):
        """
        Add column to successor matrix
        """
        self.M = np.hstack((self.M, np.zeros((len(self.M), 1))))

    def distort_bias(
        self,
        rows_to_update,
        cols_to_update,
        multipl = 3,
        max_itr = 100,
        min_per_row = 2
    ):

        # Index the region of the bias to update
        idx = np.meshgrid(
            np.arange(np.shape(self.semantic_bias)[0]),
            np.arange(np.shape(self.semantic_bias)[1])
        )
        update_region = (
            np.isin(idx[1], rows_to_update) |
            np.isin(idx[0], cols_to_update)
        )
        non_update_region = np.logical_not(update_region)
        np.fill_diagonal(update_region, False)
        np.fill_diagonal(non_update_region, False)

        # Index 1s within update region of semantic bias
        update_region_ones = update_region & (self.semantic_bias == 1)
        update_region_non_ones = update_region & (self.semantic_bias != 1)

        # Count number of 1s (ignorning identity)
        # Can think of this as biased assumptions of a transition
        n_identity = len(self.semantic_bias)
        n_ones = np.sum(self.semantic_bias == 1) - n_identity

        # How many values are already flipped
        n_flipped = np.sum((self.semantic_bias[non_update_region] == 1) & np.logical_not(self.bias[non_update_region] == 1))

        # How many still need to be flipped (in the update region)
        n_to_flip_ones = int(n_ones*(1 - self.bias_accuracy)) - n_flipped
        n_to_flip_non_ones = n_to_flip_ones*multipl
        n_update_region_ones = np.sum(update_region_ones)
        n_update_region_non_ones = np.sum(update_region_non_ones)

        # Don't flip any, too many already flipped
        if n_to_flip_ones < 0:
            n_to_flip_ones = 0
        if n_to_flip_non_ones < 0:
            n_to_flip_non_ones = 0
            
        # Run until there is at least one value per row
        for _ in range(max_itr):

            # Make update region equal to that of semantic bias
            self.bias[update_region] = self.semantic_bias[update_region]

            # Must flip the max amount if required new values exceeds number that's
            # possible to flip in the update region
            if n_update_region_ones <= n_to_flip_ones:
                n_to_flip_ones = n_update_region_ones
            if n_update_region_non_ones <= n_to_flip_non_ones:
                n_to_flip_non_ones = n_update_region_non_ones

            # Flips 0s to 1s
            samples = np.random.choice(n_update_region_non_ones, n_to_flip_non_ones, replace=False)
            to_flip = tuple(np.array(np.where(update_region_non_ones))[:, samples])
            self.bias[to_flip] = 1 - self.bias[to_flip]

            # Flips 1s to 0s
            samples = np.random.choice(n_update_region_ones, n_to_flip_ones, replace=False)
            to_flip = tuple(np.array(np.where(update_region_ones))[:, samples])
            self.bias[to_flip] = 1 - self.bias[to_flip]

            # Ensure there is at least one value per row
            if not np.all(
                (np.sum(self.bias > 0, axis=1) > min_per_row) |
                (np.sum(self.semantic_bias > 0, axis=1) <= min_per_row)
                ):
                continue
            
            break
        

        # How accurate is the final bias?
        n_match = np.sum((self.bias == 1) & (self.semantic_bias == 1)) - n_identity
        self.real_bias_accuracy = n_match/(np.sum(self.semantic_bias == 1) - n_identity)


    def compute_bias(self, start_categories, successor_categories):
        """
        Compute bias on successor matrix learning based on feature
        category match. A higher match is a greater bias

        Arguments
        ---------
        start_categories : numpy.Array
            Two-dimensional binary array of start features, where 1
            indicates a present feature and 0 indicates an absent
            feature
        successor_categories : numpy.Array
            Two-dimensional binary array of successor features, where 1
            indicates a present feature and 0 indicates an absent
            feature

        Returns
        -------
        None
        """
        
        # Compute "semantic" bias based on feature overlap
        start_categories = start_categories.astype(bool).astype(float)
        successor_categories = successor_categories.astype(bool).astype(float)
        self.semantic_bias = start_categories@successor_categories.T
        if self.conjunctive_starts and self.conjunctive_successors:
            self.semantic_bias = self.semantic_bias/self.n_per

        # Get indices for new rows and columns to update
        rows_to_update = np.arange(
            np.shape(self.bias)[0],
            np.shape(self.semantic_bias)[0]
            )
        cols_to_update = np.arange(
            np.shape(self.bias)[1],
            np.shape(self.semantic_bias)[1]
            )
        
        # Set new region of bias equal to the semantic bias
        if (np.shape(self.bias)[1] > 0) and (self.bias_accuracy != 1):
            prev_bias = self.bias.copy()
            self.bias = self.semantic_bias.copy()
            self.bias[:len(prev_bias), :len(prev_bias)] = prev_bias
        else:
            self.bias = self.semantic_bias.copy()

        # Distort bias 
        if self.bias_accuracy != 1:
            self.distort_bias(rows_to_update, cols_to_update)
        
        # Apply bias degree
        if self.segmentation < 0: # opposite to semantic structure
            self.bias = 1 - self.bias
        abs_segmentation = np.abs(self.segmentation)
        self.bias *= abs_segmentation
        self.bias += (1 - abs_segmentation)

        # Set terminal bias (make instances encode for self)        
        if self.conjunctive_starts == self.conjunctive_successors:
            self.bias_terminal = np.eye(len(self.bias))
        else:
            if self.conjunctive_successors:
                starts, successors = self.F_raw, self.S
            else:
                starts, successors = self.S, self.F_raw
            self.bias_terminal = np.array([
                np.any((start == successors) & (start != 0), axis=1)
                for start in starts
            ], dtype=float)

    def update_memory(self, state):
        """
        Adds information about the current state to memory

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array

        Returns
        -------
        None
        """

        # Track if update has occured, so we can update bias
        updated = False

        # Decompose state into features
        features_raw = self.decompose_state(state)
        if self.continuous_features:
            features = features_raw
        else:
            features = [
                self.binarize_state(feature)
                for feature in features_raw
            ]

        # Initialize memory
        if len(self.S) == 0: 
            updated = True
            self.S = np.array([state])
            self.F = np.array([features[0]])
            self.F_raw = np.array([features_raw[0]])
            self.M = np.array([[0.]])
            self.recency = np.array([0])
            self.frequency = np.array([1])
            self.F_recency = np.array([0])
            self.F_frequency = np.array([1])

        # Update state memory
        self.recency += 1
        idx = np.all(state == self.S, axis=1)
        present = np.any(idx)
        if not present:
            updated = True
            self.S = np.vstack((self.S, state))
            self.recency = np.append(self.recency, 0)
            self.frequency = np.append(self.frequency, 1)
            if self.conjunctive_starts:
                self.add_row_to_M()
            if self.conjunctive_successors:
                self.add_col_to_M() 
        else:
            self.recency[idx] = 0
            self.frequency[idx] += 1

        # Update feature memory
        for i in range(len(features)):
            feature = features[i]
            feature_raw = features_raw[i]
            idx = np.all(feature == self.F, axis=1)
            present = np.any(idx)
            if not present:
                self.F = np.vstack((self.F, feature))
                self.F_raw = np.vstack((self.F_raw, feature_raw))
                self.F_recency = np.append(self.F_recency, 0)
                self.F_frequency = np.append(self.F_frequency, 1)
                if not self.conjunctive_starts:
                    self.add_row_to_M()
                if not self.conjunctive_successors:
                    self.add_col_to_M()
            else:
                self.F_recency[idx] = 0
                self.F_frequency[idx] += 1

        # If a new state has been encountered, re-compute bias
        if updated:
            if self.conjunctive_starts:
                start_categories = self.S
            else:
                start_categories = self.F_raw
            if self.conjunctive_successors:
                successor_categories = self.S
            else:
                successor_categories = self.F_raw
            self.compute_bias(start_categories, successor_categories)

    def sample_memory(
            self,
            state,
            successors = False,
            similarity_weight_only = False
        ):
        """
        Sample weights for similar states/features in memory

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array
        successors : bool
            If True, sample weights for columns of M. 
            If False, sample weights for rows of M.
        similarity_weight_only : bool
            If True, do not use recency and frequency weights, only the
            feature similarity weight
            If False, use parameterized weights

        Returns
        -------
        p_sample : numpy.Array
            Sampler weights for rows of M
        """

        # Is this a search through conjunctive memory?
        conjunctive = (
            (successors and self.conjunctive_successors) or
            (not successors and self.conjunctive_starts)
        )

        # If not a memory sampler only retrieve exact matches in memory
        if not (self.memory_sampler and conjunctive):

            # Sample state or feature index in S or F, respectively
            if conjunctive:
                p_sample = self.get_state_index(state)
            else:
                p_sample = self.get_discrete_feature_index(state)

            # if no exact match, all are equally most similar
            no_match = np.logical_not(np.any(p_sample))
            p_sample[no_match] = 1 

        # Sample from memory
        else:

            # Get degree of feature overlap
            feature_overlap = (
                (state == self.S) &
                (self.S != 0)
            )
            state_similarity = np.sum(feature_overlap, axis=1)

            # Deal with states with all novel features
            no_overlap = np.logical_not(np.any(state_similarity))
            state_similarity[no_overlap] = 1 

            # Normalize state similarity
            state_similarity = state_similarity/np.max(state_similarity)
            
            # Get normed state recency
            state_recency = np.argsort(self.recency) + 1
            state_recency = state_recency/np.max(state_recency)

            # Get normed state frequency
            state_frequency = np.argsort(self.frequency) + 1
            state_frequency = state_frequency/np.max(state_frequency)

            # Weight similarity, recency, and frequency info
            p_sample = self.samp_weights[0]*state_similarity
            if not similarity_weight_only:
                p_sample += self.samp_weights[1]*state_recency
                p_sample += self.samp_weights[2]*state_frequency

            # Apply inhibition on extent of sampling
            p_sample = p_sample/np.max(p_sample)
            p_sample = p_sample**self.sampler_specificity      


        # Norm for similarity weights
        p_sample = p_sample/np.sum(p_sample)

        return p_sample

    def set_task(self, w_env):
        """
        Set task by mapping environment defined w onto states/features
        in memory (S/F)

        Arguments
        ---------
        w_env : numpy.Array
            Environment-defined task, which is a 1-d array reflecting
            weights on environment features

        Returns
        -------
        None
        """
        
        # If memory is empty, task is not set
        if len(self.S) == 0:
            self.w = []

        # Use environment w directly
        elif self.continuous_features and not self.conjunctive_successors:
            self.w = w_env

        # Map w onto items in memory
        else:
            self.w = self.sample_memory(w_env, successors=True)
            self.w = self.w/np.max(self.w)

    def make_action(self, actions):
        """
        Make action

        Arguments
        ---------
        actions : numpy.Array
            2-d array, where each row is a state corresponding to an
            action

        Returns
        -------
        action : int
            Index in actions for the action executed
        p : numpy.Array
            Probabilities of each action in actions
        """

        # If memory is empty, choose randomly
        if len(self.S) == 0:
            action_values = np.zeros(len(actions))

        else:

            # Evaluate states based on task
            self.V = self.M@self.w
            
            # Evaluate actions based on state evaluation
            action_values = []
            for action in actions:
                s_weight = self.sample_memory(action)
                action_values.append(self.V@s_weight)
            action_values = np.array(action_values)

        # Get choice probabilities
        if len(np.unique(action_values)) == 1:
            p = np.ones(len(action_values))/len(action_values)
        elif np.isinf(self.beta):
            p = self.p_argmax(action_values)
        else:
            p = self.softmax(action_values)

        # Sample action
        action = np.random.choice(len(p), p=p)

        return action, p

    def compose_from_set(self, feature_set, set_composition=[]):
        """
        Compose an item by choosing within a set of features, organized
        in categories. 

        Arguments
        ---------
        feature_set : numpy.Array
            A 3-d array, where the first dimension is the feature 
            categories the agent chooses within, the second dimension
            is each state, and the third dimension is each feature
        set_composition : numpy.Array
            Set a 1-d array as the composition of the set

        Returns
        -------
        composition : numpy.Array
            A 1-d array for the composed state
        p : float
            Probability of this composition
        """
        
        # Initialize empty composition and choice probabilities
        composition = np.zeros(self.n_feats, dtype=int)
        probs = []

        # Conjunctive composition constructs the set of all possible
        # compositions, and chooses within this set of conjunctions
        if self.conjunctive_composition:

            # Generate indices for all possible combinations of features
            n_features = len(feature_set)
            n_per_feature = len(feature_set[0])
            combs = np.meshgrid(*[list(range(n_per_feature))]*n_features)
            combs = np.array(combs).T.reshape(-1, n_features)

            # Construct options state set
            actions = []
            for comb in combs:
                state = np.copy(composition)
                for f, i in enumerate(comb):
                    state += feature_set[f][i]
                actions.append(state)
            actions = np.array([actions])

        # Non-conjunctive composition chooses within each feature
        # set as an independent action
        else:
            actions = feature_set

        # Make composition
        if len(set_composition) == 0:
            for options in actions:
                action, p = self.make_action(options)
                composition += options[action]
                probs.append(p[action])

        # Set composition
        else:

            # Do not decompose composition
            composition = set_composition
            if self.conjunctive_composition:
                comp_feats = [composition]
            
            # Decompose composition into features
            else:
                comp_feats = np.eye(self.n_feats, dtype=int)*composition
                comp_feats = comp_feats[np.any(comp_feats != 0, axis=1)]
            
            # Get action probabilities
            for i, options in enumerate(actions):
                action = np.all(comp_feats[i] == options, axis=1)
                p = self.make_action(options)[1]
                probs.append(p[action][0])

        # Composition probability as product of action probabilities
        p = np.prod(probs)

        return composition, p

    def p_argmax(self, values):
        """
        Get action with the highest value

        Arguments
        ---------
        values : numpy.Array
            Action values

        Returns
        -------
        p : numpy.Array
            Probabilities of each action in actions. Will be a one-hot
            vector.
        """
        p = np.eye(len(values))[np.argmax(values)]
        return p

    def softmax(self, values):
        """
        Get probabilities of actions based on a softmax function with
        an inverse temperature parameter to control how deterministic
        choice is  

        Arguments
        ---------
        values : numpy.Array
            Action values

        Returns
        -------
        p : float
            Probabilities of each action in actions
        """
        terms = np.exp(values*self.beta)

        # Account for infinite terms
        inf_terms = np.array([np.isinf(term) for term in terms])
        if np.any(inf_terms):
            terms = inf_terms

        # Account for all zero terms
        zero_terms = terms == 0
        if np.all(zero_terms):
            terms = zero_terms

        # Get probabilities
        p = terms/np.sum(terms)

        # Deal with both 0 probabilities to large terms
        if np.sum(p) == 0:
            p = np.eye(len(p))[np.argmax(terms)]
    
        return p

    def get_state_index(self, state):
        """
        Get state index in state memory (S)

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array

        Returns
        -------
        idx : numpy.Array
            Bool array index for state in S
        """
        idx = np.all(state == self.S, axis=1)
        return idx
    
    def get_discrete_feature_index(self, state):
        """
        Get indices for features of state in feature memory (F)

        Arguments
        ---------
        state : numpy.Array
            One-dimensional state array

        Returns
        -------
        idx : numpy.Array
            Bool array index for features of state in F
        """
        features = self.binarize_state(state)
        features = self.decompose_state(features)
        idx = np.zeros(len(self.F))
        for feature in features:
            idx += np.all(feature == self.F, axis=1)
        idx = idx.astype(bool)
        return idx
    
    def get_M_update_weights(self, state, state_new):
        """
        Get weights on rows of M for the present and successor states

        Arguments
        ---------
        state : numpy.Array
            One-dimensional current state array
        state_new : numpy.Array
            One-dimensional successor state array

        Returns
        -------
        s_weight : numpy.Array
            Weight for present state
        s_new_weight : numpy.Array
            Weight for successor state
        """
        if self.conjunctive_starts:
            s_weight = self.get_state_index(state)
            s_new_weight = self.get_state_index(state_new)
        elif not self.continuous_features:
            s_weight = self.get_discrete_feature_index(state)
            s_new_weight = self.get_discrete_feature_index(state_new)
        else:
            s_weight = state
            s_new_weight = state_new
        s_weight = s_weight.reshape(-1, 1)
        return s_weight, s_new_weight
    
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
    
    def weight_bias_matrix(self, weight, bias):
        """
        Weight and normalize bias matrix

        Arguments
        ---------
        weight : numpy.Array
            One-dimensional weight array
        bias : numpy.Array
            Two-dimensional bias matrix

        Returns
        -------
        bias : numpy.Array
            Weighted and normalized bias matrix
        """
        bias = weight*bias
        norm = np.sum(bias, axis=1)
        norm[norm == 0] = 1
        bias = bias/norm
        return bias
    
    def decay_alpha(self):
        """
        Decay learning rate based on state visitation frequency

        Returns
        -------
        alpha : float
            Decayed learning rate
        """
        if self.conjunctive_starts:
            frequency = self.frequency
        else:
            frequency = self.F_frequency
        alpha = self.alpha*frequency**-self.alpha_decay
        return alpha

    def update_M(self, state, state_new):
        """
        Update successor matrix (M)

        Arguments
        ---------
        state : numpy.Array
            One-dimensional current state array
        state : numpy.Array
            One-dimensional successor state array

        Returns
        -------
        None
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

