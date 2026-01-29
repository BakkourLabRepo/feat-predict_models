
import numpy as np
from itertools import permutations

def get_instance_combinations(instances, n_per):
    """
    Get all possible instance combinations

    Arguments
    ---------
    instances : list
        List of feature instances
    n_per : int
        Number of features per state
    
    Returns
    -------
    combs : numpy.Array
        Array of all possible instance combinations
    """
    combs = np.meshgrid(*[instances]*n_per)
    combs = np.array(combs).T.reshape(-1, n_per)
    return combs

class Env:
    """
    Environment class for multi-feature state space

    Attributes
    ----------
    tmat : numpy.Array
        Transition matrix for feature instances
    n_feats : int
        Number of features in the environment
    n_fixed : int
        Number of fixed features in each state
    n_per : int
        Number of features per state
    start_insts : list
        List of starting feature instances
    r : numpy.Array
        Reward matrix for feature instances
    feat_tmat : numpy.Array
        Transition matrix for features
    max_steps : int
        Maximum number of steps per episode
    pr : list
        List of probabilities for each reward matrix
    probabilistic : bool
        If True, rewards are probabilistic. If False, rewards are
        deterministic.
    rel_cross_feature_inst_freq : int
        Relative frequency of instances that are the same across
        features when sampling states during transition training
    continuous_features : bool
        If True, features are continuous.
        If False, features are discrete.
    """

    def __init__(
        self,
        tmat,
        n_feats,
        n_fixed,
        n_per,
        start_insts,
        r,
        max_steps = None,
        feat_tmat = [],
        pr = [1],
        probabilistic = False,
        rel_cross_feature_inst_freq = 1,
        continuous_features = False
    ):
        self.tmat = tmat
        if len(feat_tmat) == 0:
            feat_tmat = np.eye(n_feats)
        self.feat_tmat = feat_tmat
        self.n_feats = n_feats
        self.n_fixed = n_fixed
        self.n_per = n_per
        self.insts = np.vstack((
            start_insts, # start instances
            np.where(np.diag(tmat) == 1)[0] + 1 # terminal instances
            ))
        self.r = r
        if max_steps is None:
            self.max_steps = len(tmat)
        else:
            self.max_steps = max_steps
        self.pr = pr
        self.probabilistic = probabilistic
        self.rel_cross_feature_inst_freq = rel_cross_feature_inst_freq
        self.continuous_features = continuous_features

        self.check_features()
        self.get_likely_outcomes()
        self.gen_start_states()

    def sample_cat_combination(self):
        """
        Sample a category combination

        Returns
        -------
        comb_sample : list
            Sampled category combination
        """
        combs = list(self.states.keys())
        comb_sample = combs[np.random.choice(len(combs))]
        return comb_sample

    def sample_features(self, comb=[], step=0):
        """
        Sample features for action selection

        Arguments
        ---------
        comb : list
            List indicating which categories are present (1) or
            absent (0)
        step : int
            Step index for instances. Use -1 for terminal instances
        """

        # Sample category combination if not provided
        if len(comb) == 0:
            comb = self.sample_cat_combination()

        # Select instances at the specified step
        insts = np.copy(self.insts[step])

        # Number of instances and categories
        n_insts = len(insts)
        n_cats = np.sum(comb)

        # Action matrix with unique row for each instance x category
        comb = np.array(comb)
        comb = np.eye(len(comb), dtype=int)[comb.astype(bool)]
        comb = np.repeat(comb, n_insts, axis=0)
        comb = comb.reshape(n_cats, n_insts, -1)
        self.a = comb*insts.reshape(-1, 1)

    def get_successor(
            self,
            s,
            most_likely = False,
            invert = False,
            use_feat_tmat = True
            ):
        """
        Get successor state for a given state

        Arguments
        ---------
        s : numpy.Array
            State to get successor for
        most_likely : bool
            If True, use most likely transitions. If False, sample
            transitions probabilistically.
        invert : bool
            If True, get predecessor state. If False, get successor
            state.
        use_feat_tmat : bool
            If True, use feature transition matrix to select feature
            to update. If False, update features in order.
        """

        if most_likely:
            tmat = np.copy(self.lik_tmat)
        else:
            tmat = np.copy(self.tmat)
        if invert:
            tmat = tmat.T
            np.fill_diagonal(tmat, 0) # avoid absorbing states
            terminal_insts = self.insts[0]
        else:
            terminal_insts = self.insts[-1]
        
        # Terminal states are absorbing
        if np.any(np.isin(s, terminal_insts)):  
            s_new = np.copy(s)

        # Use transition matrices to update non-terminal states
        else:          
            s_new = np.zeros(len(s), dtype=int)
            for feat in range(len(s)):
                if s[feat] != 0:
                    inst = s[feat] - 1
                    if use_feat_tmat:
                        feat_new = np.random.choice(
                            np.arange(len(self.feat_tmat)),
                            p = self.feat_tmat[feat]
                            )
                    else:
                        feat_new = feat
                    s_new[feat_new] = np.random.choice(
                        np.arange(len(tmat)) + 1,
                        p = tmat[inst]
                        )
        
        return s_new
    
    def get_start_state(self, s, step=0):
        """
        Get start state that precedes a given state

        Arguments
        ---------
        s : numpy.Array
            State to get predecessor for
        step : int
            Step index for start instances
        
        Returns
        -------
        start_state : numpy.Array
            Start state that precedes given state
        """
        pred = np.copy(s)
        while not np.any(np.isin(pred, self.insts[step])):
            pred = self.get_successor(pred, most_likely=True, invert=True)
        return pred
        
    def step(self):
        """
        Step the environment
        """

        # Step each feature based on the transition matrix
        self.s_new = self.get_successor(self.s)

        # Get immediate reward (for state you're leaving)
        self.reward = self.get_reward(self.s)

    def update_current_state(self):
        """
        Mark "new" state as current
        """
        self.s = np.copy(self.s_new)

    def get_reward(self, s, most_likely=False):
        """
        Get reward for given state
        
        Arguments
        ---------
        s : numpy.Array
            State to get reward for
        most_likely : bool
            If True, use most likely rewards. If False, sample
            rewards probabilistically.
        
        Returns
        -------
        reward : float
            Reward for given state
        """
        if most_likely:
            feat_rewards = [self.lik_r[f] for f in (s[s > 0] - 1)]
        else:
            feat_rewards = [
                self.r[np.random.choice(len(self.pr), p=self.pr), f]
                for f in (s[s > 0] - 1)
                ]
            
        if self.probabilistic:
            p_reward = np.mean(feat_rewards)
            if most_likely:
                reward = p_reward
            else:
                reward = np.random.choice(
                    [0, 1], 1,p=[1 - p_reward, p_reward])[0]
        else:
            reward = np.sum(feat_rewards)
            
        return reward

    def check_absorbing(self):
        """
        Check if any absorbing feature transition has occured
        """
        if np.any((self.s_new == self.s) & (self.s_new != 0)):
            return True
        else:
            return False

    def get_likely_outcomes(self):
        """
        Calculate most probable transition matrix
        """

        # Most likely transitions
        self.lik_tmat = np.zeros(np.shape(self.tmat))
        self.lik_tmat[
            np.arange(len(self.tmat)),
            np.argmax(self.tmat, axis=1)
            ] = 1

        # Most likely rewards
        self.lik_r = self.r[np.argmax(self.pr)]

    def assign_insts_to_cats(self, cat_comb, inst_combs):
        """
        Assign instance combinations to category combinations

        Arguments
        ---------
        cat_comb : numpy.Array
            Array indicating which categories are present (1) or
            absent (0)
        inst_combs : numpy.Array
            Array of instance combinations to assign to categories

        Returns
        -------
        states : numpy.Array
            Array of states with instances assigned to categories
        """
        states = np.tile(cat_comb, (len(inst_combs), 1))
        states[states.astype(bool)] = inst_combs.flatten()
        return states

    def gen_start_states(self):
        """
        Generate all possible start states
        """

        # Number of non-fixed features present
        n_present = self.n_per - self.n_fixed

        # Get non-fixed feature combinations
        n_absent = self.n_feats - self.n_fixed - n_present
        f_present = [1]*n_present + [0]*n_absent
        self.combs = np.unique(
            np.array(list(permutations(f_present))),
            axis = 0
            )

        # Add fixed states
        f_fixed = np.ones((len(self.combs), self.n_fixed))
        self.combs = np.hstack((f_fixed, self.combs)).astype(int)

        # Get start and terminal instance combinations
        start_combs = np.array([
            get_instance_combinations(start_insts, self.n_per)
            for start_insts in self.insts[:-1]
            ])
        terminal_combs = get_instance_combinations(self.insts[-1], self.n_per)
        successor_combs = [
            self.get_successor(s, most_likely=True, use_feat_tmat=False)
            for s in start_combs[0]
            ]
        successor_combs = np.array(successor_combs)

        # Create a dictionary of all start states for each combination
        self.states = {}
        for comb in self.combs:
            self.states[tuple(comb)] = [
                    self.assign_insts_to_cats(comb, step_start_combs)
                    for step_start_combs in start_combs
                ]
            self.states[tuple(comb)] += [
                self.assign_insts_to_cats(comb, terminal_combs)
                ]


    def check_features(self):
        """
        Check valid feature numbers have been provided
        """

        if self.n_feats < self.n_per:
            print("""
            WARNING: More features per state than total features in the environment
            Setting: features per state = total features
            """)
            self.n_per = self.n_feats

        if self.n_per < self.n_fixed:
            print("""
            WARNING: More fixed features than features per state
            Setting: fixed features = features per state
            """)
            self.n_fixed = self.n_per

        elif self.n_feats < self.n_fixed:
            print("""
            WARNING: More fixed features than total features in the environment
            Setting: fixed features = total features
            """)
            self.n_fixed = self.n_feats

    def check_terminal(self, state):
        """
        Check if a state is terminal or not

        Arguments
        ---------
        state : numpy.Array
            Array representing a state
        
        Returns
        -------
        is_terminal : bool
            Returns True if state is terminal. False otherwise.
        """
        is_terminal = np.any(np.isin(self.insts[-1], state))
        return is_terminal
    
    def check_step(self, state):
        """
        Check step index of a state

        Arguments
        ---------
        state : numpy.Array
            Array representing a state
        
        Returns
        -------
        step : int
            Step index of the state
        """
        for step in range(len(self.insts)):
            if np.any(np.isin(self.insts[step], state)):
                return step
        return step