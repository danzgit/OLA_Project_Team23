import numpy as np


class Environment:
    def __init__(self, n_arms, n_customers, margin1, conv_rate1):
        self.n_arms = n_arms  # 4 arms
        self.n_customers = n_customers
        self.margin1 = margin1
        self.conv_rate1 = conv_rate1

    def margin_mode(self, pulled_arm, weight_edge):
        tot_margin = 0
        for cust_class in range(len(self.n_customers)):  # 3 cust class
            customers = np.random.binomial(self.n_customers[cust_class] * weight_edge, self.conv_rate1[
                cust_class, pulled_arm])  # binomial in range of n_customers of selected cust class
            tot_margin += self.margin1[pulled_arm] * customers
        return tot_margin

    def reward_mode(self, pulled_arm, weight_edge):  # ROUND METHOD
        for cust_class in range(len(self.n_customers)):
            reward = np.random.binomial(1, weight_edge * self.conv_rate1[
                cust_class, pulled_arm])  # binomial probabilities using np used for reward value, 1 or 0
        return reward

    def margin_agg_mode(self, pulled_arm, weight_edge):
        tot_margin = 0
        customers = np.random.binomial(self.n_customers * weight_edge, self.conv_rate1[
            pulled_arm])  # binomial in range of n_customers of selected cust class
        tot_margin += self.margin1[pulled_arm] * customers
        return tot_margin

    def reward_agg_mode(self, pulled_arm, weight_edge):  # ROUND METHOD
        reward = np.random.binomial(1, weight_edge * self.conv_rate1[
            pulled_arm])  # binomial probabilities using np used for reward value, 1 or 0
        return reward
