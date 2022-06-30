import numpy as np
import matplotlib.pyplot as plt
# import classes that we have defined
from Environment import *
from TS_Learner import *
from Greedy_Learner import *
from UCB_Learner import *

#################### Initiation ####################

n_arms = 4  # number of candidates/prices
n_prods = 5

# margin
prod1_price1 = np.array([400, 450, 500, 550])  # bernoulli distribution for reward function of the arms
prod1_margin1 = np.array([50., 100., 150., 200.])
n_customers = np.array([634, 521, 869])  # class 1, 2, and 3 respectively

conv_rate1 = np.array([[0.55, 0.5, 0.51, 0.47],
                       [0.42, 0.45, 0.35, 0.27],
                       [0.65, 0.67, 0.7, 0.55]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod2_price2 = np.array([100, 250, 300, 550])  # bernoulli distribution for reward function of the arms
prod2_margin2 = np.array([50., 150., 200., 450.])

conv_rate2 = np.array([[0.35, 0.53, 0.15, 0.67],
                       [0.62, 0.35, 0.45, 0.87],
                       [0.25, 0.87, 0.5, 0.25]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod3_price3 = np.array([200, 250, 400, 450])  # bernoulli distribution for reward function of the arms
prod3_margin3 = np.array([100., 150., 200., 250.])

conv_rate3 = np.array([[0.15, 0.54, 0.56, 0.37],
                       [0.47, 0.25, 0.55, 0.24],
                       [0.35, 0.27, 0.57, 0.65]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod4_price4 = np.array([30, 60, 110, 210])  # bernoulli distribution for reward function of the arms
prod4_margin4 = np.array([20., 50., 100., 200.])

conv_rate4 = np.array([[0.35, 0.56, 0.31, 0.57],
                       [0.49, 0.48, 0.05, 0.17],
                       [0.15, 0.37, 0.57, 0.85]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod5_price5 = np.array([70, 130, 200, 530])  # bernoulli distribution for reward function of the arms
prod5_margin5 = np.array([40., 100., 170., 450.])
n_customers = np.array([634, 521, 869])  # class 1, 2, and 3 respectively

conv_rate5 = np.array([[0.55, 0.5, 0.51, 0.47],
                       [0.42, 0.45, 0.35, 0.27],
                       [0.65, 0.67, 0.7, 0.55]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

# optimum
opt = np.max(conv_rate1)

T = 365  # simulation time horizon
n_experiments = 2000  # at least 1000 to make less noise

ts_reward_per_experiment = []
ts_margin_per_experiment = []

ucb_reward_per_experiment = []
ucb_margin_per_experiment = []

gr_reward_per_experiment = []
gr_margin_per_experiment = []

ts_reward_per_product = []
ts_margin_per_product = []

ucb_reward_per_product = []
ucb_margin_per_product = []

gr_reward_per_product = []
gr_margin_per_product = []

products_margin = [prod1_margin1, prod2_margin2, prod3_margin3, prod4_margin4, prod5_margin5]
products_conversion = [conv_rate1, conv_rate2, conv_rate3, conv_rate4, conv_rate5]
products_prices = [prod1_price1, prod2_price2, prod3_price3, prod4_price4, prod5_price5]

#################### Execution ####################
for i in range(n_experiments):
    for j in range(n_prods):
        env = Environment(n_arms, n_customers, products_margin[j], products_conversion[j])
        ts_learner = TS_Learner(n_arms)
        ucb_learner = UCB_Learner(n_arms)
        gr_learner = Greedy_Learner(n_arms)

        for t in range(T):  # learner and environment interaction during the duration of T=365 (a year)
            # Thompson Sampling Learner
            ts_arm_pulled = ts_learner.pull_arm()
            ts_margin = env.margin_mode(ts_arm_pulled)
            ts_reward = env.reward_mode(ts_arm_pulled)
            ts_learner.update(ts_arm_pulled, ts_reward, ts_margin)

            # UCB1 Learner
            ucb_arm_pulled = ucb_learner.pull_arm()
            ucb_margin = env.margin_mode(ucb_arm_pulled)
            ucb_reward = env.reward_mode(ucb_arm_pulled)
            ucb_learner.update(ucb_arm_pulled, ucb_reward, ucb_margin)

            # Greedy Learner
            gr_arm_pulled = gr_learner.pull_arm()
            gr_margin = env.margin_mode(gr_arm_pulled)
            gr_reward = env.reward_mode(gr_arm_pulled)
            gr_learner.update(gr_arm_pulled, gr_reward, gr_margin)

        ts_reward_per_product.append(ts_learner.collected_rewards)
        ucb_reward_per_product.append(ucb_learner.collected_rewards)
        ts_margin_per_product.append(ts_learner.collected_margins)
        gr_margin_per_product.append(gr_learner.collected_margins)

    # for algorithm performance (Bandit comparison)
    ts_reward_per_experiment.append(ts_reward_per_product)
    ucb_reward_per_experiment.append(ucb_reward_per_product)

    # for margin performance
    ts_margin_per_experiment.append(ts_margin_per_product)
    gr_margin_per_experiment.append(gr_margin_per_product)

ts_margin_per_experiment_transposed = list(map(list, zip(*ts_margin_per_experiment)))
gr_margin_per_experiment_transposed = list(map(list, zip(*gr_margin_per_experiment)))
ts_reward_per_experiment_transposed = list(map(list, zip(*ts_reward_per_experiment)))
ucb_reward_per_experiment_transposed = list(map(list, zip(*ucb_reward_per_experiment)))

#################### Algorithm Performance (Regret) ####################
for i in range(n_prods):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Regret over time - Product ' + str(i+1))
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(opt - ts_reward_per_experiment_transposed[i], axis=0)), 'g')
    plt.plot(np.cumsum(np.mean(opt - ucb_reward_per_experiment_transposed[i], axis=0)), 'b')
    plt.legend(['Thompson Sampling', 'UCB1'], loc='lower right')
    plt.show()

    #################### Margin Performance ####################
    print('MARGIN PERFORMANCE')
    print()
    print("Thompson Sampling Algorithm")
    print(
        "Selected price for product " + str(i+1) + " :  [{}]".format(products_prices[i][np.argmax([len(a) for a in ts_learner.rewards_per_arm])]))
    print('Total margin:  [{}]'.format(np.sum(ts_margin_per_experiment_transposed[i])))
    print()
    print("Greedy Algorithm")
    print(
      "Selected price for product " + str(i+1) + " :  [{}]".format(products_prices[i][np.argmax([len(a) for a in gr_learner.rewards_per_arm])]))
    print('Total margin:  [{}]'.format(np.sum(gr_margin_per_experiment_transposed[i])))
    print()

    plt.figure(figsize=(8, 6), dpi=80)
    plt.title('Cumulative Margin')
    plt.xlabel("Days")
    plt.ylabel("Margin")
    plt.plot(np.cumsum(ts_margin_per_experiment_transposed[i]), 'g', label='Thompson Sampling')
    plt.plot(np.cumsum(gr_margin_per_experiment_transposed[i]), 'r', label='Greedy')
    plt.legend(['Thompson Sampling', 'Greedy'], loc='lower right')
    plt.show()
