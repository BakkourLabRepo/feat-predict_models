
import numpy as np
from scipy.stats import rankdata
from itertools import permutations
from feat_predict.modelling.helper_funcs import (
    assign_insts_to_cats,
    id_from_feats,
    sample_from_array,
    sample_row_except
)

### Environment class ##########################################################

class Env:

    """
    Task environment. Tracks and updates state and feature values.
    :param tmat: feature transition matrix (same for all features)
    :param n_feats: number of feature categories
    :param n_fixed: number of fixed features that occur in every state
    :param n_per: number of features per state
    :param start_insts: array of start instance IDs
    :param r: possible absolute reward values for each instance (column is
        instance, row is possible values instance can have)
    :param pr: probability of possible value for each instance in r
    rel_cross_feature_inst_freq : relative frequence of states that
    have matching instance values across features during transition
    training
    p_ttrain_probes : probability that the target feature
    combination during transition training shares only one feature
    with the action states
    """

    def __init__(
        self,
        tmat,
        n_feats,
        n_fixed,
        n_per,
        start_insts,
        r,
        pr=[1],
        probabilistic=False,
        rel_cross_feature_inst_freq=1,
        continuous_features = False
    ):
        self.tmat = tmat
        self.n_feats = n_feats
        self.n_fixed = n_fixed
        self.n_per = n_per
        self.start_insts = start_insts
        self.terminal_insts = np.where(np.diag(tmat) == 1)[0] + 1
        self.r = r
        self.pr = pr
        self.probabilistic = probabilistic
        self.rel_cross_feature_inst_freq = rel_cross_feature_inst_freq
        self.continuous_features = continuous_features

        self.check_features()
        self.get_likely_outcomes()
        self.gen_start_states()

    def sample_cat_combination(self):
        """
        Sample a category combination based on possible states
        """
        return list(self.states.keys())[np.random.choice(len(self.states.keys()))]

    def sample_features(self, comb=[], terminal=False):
        """
        Sample features for composition
        :param comb: if set, sample from a specific category combination
        :param terminal: if True, sample terminal states. If False, sample start
            states
        """

        # Sample category combination if not provided
        if len(comb) == 0:
            comb = self.sample_cat_combination()

        # Get terminal or start instances
        insts = np.copy([self.start_insts, self.terminal_insts][terminal])

        # Number of instances and categories
        n_insts = len(insts)
        n_cats = np.sum(comb)

        # Construct action matrix with unique row for each instance x category
        comb = np.array(comb)
        comb = np.repeat(np.eye(len(comb), dtype=int)[comb.astype(bool)], n_insts, axis=0).reshape(n_cats, n_insts, -1)
        self.a = comb*insts.reshape(-1, 1)


    def sample_actions(self, comb=[], terminal=False, n_actions=2,
                       transitions=False, feature_overlap=0, probe=False):
        """
        Sample actions from the state space. Also sample target for transition training
        :param comb: if set, sample from a specific category combination
        :param terminal: if True, sample terminal states. If False, sample start
            states
        :param n_actions: number of actions to sample (if features == False)
        :param transitions: if True, sample for transition training. Actions will have
            different successors. if False, sample for reward-based choices. Actions
            will have different (successor) values
        :param feature_overlap: if -1, must be no overlap between features of
            action states. if 1, must be overlap between feature of action
            states. If 0, no constraint is set.
        :param probe: if True, run transition training probe trial
        """

        # Sample category combination if not provided
        self.comb = comb
        if len(self.comb) == 0:
            self.comb = self.sample_cat_combination()
        
        # Target feature combination is usually same as for actions
        self.target_comb = self.comb

        # Sample probabilities differ for transition training
        if transitions:

            # Sample for probe trial
            if probe:

                # sample target that has only 1 matching feature with
                # the current transition training combination
                combs = np.array(list(self.states.keys()))
                comb_overlap = (
                    (self.comb == combs) &
                    (self.comb != 0)
                )
                freq_comb_overlap = np.sum(comb_overlap, axis=1)
                idx = freq_comb_overlap == 1
                self.target_comb = sample_from_array(combs[idx], 1)[0]

                # Re-define action feature combination so that actions
                # do no have overlapping features with the target
                comb_overlap = (
                    (self.target_comb == combs) &
                    (self.target_comb != 0)
                )
                freq_comb_overlap = np.sum(comb_overlap, axis=1)
                idx = freq_comb_overlap == 0
                self.comb = sample_from_array(combs[idx], 1)[0]                

            # Action sample probabilities
            p_sample = self.states[tuple(self.comb)]['p_sample']
 
            # Get target state set
            target_set = self.states[tuple(self.target_comb)]['likely_successor']

         # Get state sets
        pos = ['start', 'terminal'][terminal]
        action_set = self.states[tuple(self.comb)][pos]

        if not transitions:
            # Action sample probabilities
            p_sample = np.ones(len(action_set))
            p_sample = p_sample/np.sum(p_sample)

        # Loop until valid action combination sampled
        while True:

            # Sample for matched target
            if feature_overlap:

                # Sample first action
                self.a = sample_from_array(action_set, 1, p=p_sample)

                # Sample additional actions
                for _ in range(n_actions - 1):

                    # Index for completely non-overlapping states
                    if feature_overlap == -1:
                        idx = np.all(
                            (action_set != self.a[0]) | (np.array(self.comb) == 0),
                            axis=1
                        )

                    # Index for at least partially overlapping states
                    if feature_overlap == 1:
                        idx = np.any(
                            (action_set == self.a[0]) & (np.array(self.comb) != 0),
                            axis=1
                        )

                    # Sample from subset of possible states
                    p_sample_action = p_sample*idx
                    p_sample_action = p_sample_action/np.sum(p_sample_action)
                    self.a = np.append(
                        self.a,
                        sample_from_array(action_set, 1, p=p_sample),
                        axis=0
                    )

            # Sample for non-matched target
            else:
                self.a = sample_from_array(action_set, n_actions, p=p_sample)

            # Sample target if transition training
            if transitions:
                if self.sample_target(
                    feature_overlap = feature_overlap,
                    s_set = target_set,
                    action_set = action_set
                ):
                    continue # continue loop if invalid target sampled
            else: # blank target
                self.target = np.array([[0]*self.n_feats])

            # Check outcomes are different
            self.a = self.a.reshape(1, n_actions, -1)
            if self.check_unequal_outcomes(transitions=transitions):
                break

    
    def sample_target(
            self,
            feature_overlap = 0,
            s_set = None,
            action_set = None
        ):
        """
        Sample a target. Returns True when an invalid target is sampled
        :param feature_overlap: if -1, target is a direct successor of an action
            state. if 0, target does not have to be a direct successor
        :param s_set: set of states to sample from. if feature_overlap == False,
            this must be set
        """

        # Sampling when evaluating just one action item
        if len(self.a) == 1:
            
            # Coin flip if target is a direct successor or not 
            if np.random.rand() < .5: # direct
                start_item = self.a[0]
            else: # non-direct
                start_item = sample_row_except(action_set, self.a[0])

            # Target is successor of sampled item
            self.target = self.get_successor(start_item, most_likely=True)
            self.target = np.array([self.target])

        elif feature_overlap == -1:

            # Target is the most likley direct successor of one action state
            start_item = sample_from_array(self.a)[0]
            self.target = np.array([self.get_successor(start_item, most_likely=True)])

            # Check no target features overlap with action state features
            if np.any((self.target == self.a) & (self.target != 0)):
                return True

        else:

            # Find likely successors with no features in the action states
            poss_targets = np.array([np.all((s != self.a) | (s == 0)) for s in s_set])
            p_target = poss_targets/np.sum(poss_targets)

            # Target is a random sample from these successors
            self.target = sample_from_array(s_set, n=1, p=p_target)

        return False
    
    def convert_actions_for_target(self):
        # Convert actions to equivalent states for target comb 
        target_actions = []
        for act in self.a[0]:
            action_states = self.states[tuple(self.comb)]['start']
            idx = np.all(action_states == act, axis=1)
            target_act = self.states[tuple(self.target_comb)]['start'][idx]
            target_actions.append(target_act[0])
        target_actions = np.array(target_actions)
        return target_actions

    def check_unequal_outcomes(self, transitions=False):
        """
        If actions lead to the same goal outcome, return False. Otherwise,
            return True.
        :param transitions: if 1, base on reward values. If 2, base on successors'
            overlap with the target
        """

        # If only one action, no need to check
        if len(self.a[0]) == 1:
            return True

        # Based on likely successor's overlap with target
        if transitions:
            target_actions = self.convert_actions_for_target()
            a_vals = [np.sum(self.get_successor(act, most_likely=True) == self.target)
                      for act in target_actions]

        # Based on likely reward values
        else:
            a_vals = np.array([self.get_reward(self.get_successor(act, most_likely=True), most_likely=True) for act in self.a[0]])

        if len(np.unique(a_vals)) > 1:
            return True
        return False

    def get_successor(self, s, most_likely=False):
        """
        Get successor to state
        :param s: state to get successor for
        :param most_likely: if True, get the most likely successor. if False,
            get successor based on raw transition matrix
        """
        if most_likely:
            tmat = self.lik_tmat
        else:
            tmat = self.tmat
        s_new = np.copy(s)
        if not np.any(np.isin(s, self.terminal_insts)):
            for feat in range(len(s)):
                if s[feat] != 0:
                    s_new[feat] = np.random.choice(np.arange(len(tmat)) + 1, p=tmat[s[feat] - 1])
        return s_new

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
        :param s: state to get reward for
        """
        if most_likely:
            feat_rewards = [self.lik_r[f] for f in (s[s > 0] - 1)]
        else:
            feat_rewards = [self.r[np.random.choice(len(self.pr), p=self.pr), f]
                            for f in (s[s > 0] - 1)]
            
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
        self.lik_tmat[np.arange(len(self.tmat)), np.argmax(self.tmat, axis=1)] = 1

        # Most likely rewards
        self.lik_r = self.r[np.argmax(self.pr)]

    def gen_start_states(self):
        """
        Generate dictionary of start states indexed by feature
        combinations
        """

        # Number of non-fixed features present
        n_present = self.n_per - self.n_fixed

        # Get non-fixed feature combinations
        n_absent = self.n_feats - self.n_fixed - n_present
        f_present = [1]*n_present + [0]*n_absent
        self.combs = np.unique(np.array(list(permutations(f_present))), axis=0)

        # Add fixed states
        f_fixed = np.ones((len(self.combs), self.n_fixed))
        self.combs = np.hstack((f_fixed, self.combs)).astype(int)

        # Get start and terminal instance combinations
        start_combs = np.meshgrid(*[self.start_insts]*self.n_per)
        start_combs = np.array(start_combs).T.reshape(-1, self.n_per)
        terminal_combs = np.meshgrid(*[self.terminal_insts]*self.n_per)
        terminal_combs = np.array(terminal_combs).T.reshape(-1, self.n_per)
        successor_combs = [self.get_successor(s, most_likely=True)
                           for s in start_combs]
        successor_combs = np.array(successor_combs)

        # Create a dictionary of all start states for each combination
        self.states = {}
        for comb in self.combs:

            # Create states
            self.states[tuple(comb)] = {
                'start': assign_insts_to_cats(comb, start_combs),
                'terminal': assign_insts_to_cats(comb, terminal_combs),
                'likely_successor': assign_insts_to_cats(comb, successor_combs)
                }
            

            # Get probability of sampling states (during transition
            # training) based on relative frequency of states with
            # matching instance levels across-features

            # Get states with matching instance levels
            matching_inst_levels = []
            for s in self.states[tuple(comb)]['start']:
                inst_levels = np.unique(s)
                inst_levels = inst_levels[inst_levels != 0]
                matching_inst_levels.append(len(inst_levels) == 1)
            matching_inst_levels = np.array(matching_inst_levels)

            # Convert frequencies into probabilities
            freq = 1 + matching_inst_levels*(self.rel_cross_feature_inst_freq - 1)
            self.states[tuple(comb)]['p_sample'] = freq/np.sum(freq)


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
        is_terminal = np.any(np.isin(self.terminal_insts, state))
        return is_terminal