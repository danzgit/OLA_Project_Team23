import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# import classes that we have defined
from Environment_combined import *
from TS_Learner import *
from Greedy_Learner import *
from UCB_Learner import *


def weighted_adjmatrix(adjlist, nodes):
    '''Returns a (weighted) adjacency matrix as a NumPy array.'''
    matrix = []
    for node in nodes:
        weights = {endnode: float(weight)
                   for w in adjlist.get(node, {})
                   for endnode, weight in w.items()}
        matrix.append([weights.get(endnode, 0) for endnode in nodes])
    matrix = np.array(matrix)
    return matrix + matrix.transpose()


#################### Initiation ####################

n_arms = 4  # number of candidates/prices
n_prods = 5

# margin
prod1_price1 = np.array([400, 450, 500, 550])  # bernoulli distribution for reward function of the arms
prod1_margin1 = np.array([50., 100., 150., 200.])
n_customers1 = np.array([634, 521, 869])  # class 1, 2, and 3 respectively
# P1_weights = {1: 0.0, 2: 0.2, 3: 0.1, 4: 0.4, 5: 0.3}
conv_rate1 = np.array([[0.55, 0.5, 0.51, 0.47],
                       [0.42, 0.45, 0.35, 0.27],
                       [0.65, 0.67, 0.7, 0.55]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod2_price2 = np.array([100, 250, 300, 550])  # bernoulli distribution for reward function of the arms
prod2_margin2 = np.array([50., 150., 200., 450.])
n_customers2 = np.array([334, 621, 369])
# P2_weights = {1: 0.3, 2: 0.0, 3: 0.3, 4: 0.25, 5: 0.15}
conv_rate2 = np.array([[0.35, 0.53, 0.15, 0.67],
                       [0.62, 0.35, 0.45, 0.87],
                       [0.25, 0.87, 0.5, 0.25]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod3_price3 = np.array([200, 250, 400, 450])  # bernoulli distribution for reward function of the arms
prod3_margin3 = np.array([100., 150., 200., 250.])
n_customers3 = np.array([124, 641, 909])
# P3_weights = {1: 0.34, 2: 0.16, 3: 0.0, 4: 0.4, 5: 0.1}
conv_rate3 = np.array([[0.15, 0.54, 0.56, 0.37],
                       [0.47, 0.25, 0.55, 0.24],
                       [0.35, 0.27, 0.57, 0.65]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod4_price4 = np.array([30, 60, 110, 210])  # bernoulli distribution for reward function of the arms
prod4_margin4 = np.array([20., 50., 100., 200.])
n_customers4 = np.array([884, 225, 579])
# P4_weights = {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.0, 5: 0.1}
conv_rate4 = np.array([[0.35, 0.56, 0.31, 0.57],
                       [0.49, 0.48, 0.05, 0.17],
                       [0.15, 0.37, 0.57, 0.85]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

prod5_price5 = np.array([70, 130, 200, 530])  # bernoulli distribution for reward function of the arms
prod5_margin5 = np.array([40., 100., 170., 450.])
n_customers5 = np.array([934, 427, 434])  # class 1, 2, and 3 respectively
# P5_weights = {1: 0.24, 2: 0.35, 3: 0.01, 4: 0.4, 5: 0.0}
conv_rate5 = np.array([[0.55, 0.5, 0.51, 0.47],
                       [0.42, 0.45, 0.35, 0.27],
                       [0.65, 0.67, 0.7, 0.55]])  # 3 class, 4prices=4arms=4conv_rate, for only 1 product

g = {'1': [{'2': '0.2'}, {'3': '0.4'}, {'4': '0.4'}, {'5': '0.3'}],
     '2': [{'1': '0.3'}, {'3': '0.3'}, {'4': '0.25'}, {'5': '0.15'}],
     '3': [{'1': '0.34'}, {'2': '0.16'}, {'4': '0.4'}, {'5': '0.1'}],
     '4': [{'1': '0.6'}, {'2': '0.2'}, {'3': '0.1'}, {'4': '0.1'}],
     '5': [{'1': '0.24'}, {'2': '0.35'}, {'3': '0.01'}, {'4': '0.4'}]}

adjacent_matrix = weighted_adjmatrix(g, list('12345'))

# print(adjacent_matrix)
T = 365  # simulation time horizon
n_experiments = 10

ts_reward_per_experiment = []
ts_margin_per_experiment = []

ucb_reward_per_experiment = []
ucb_margin_per_experiment = []

ts_reward_agg_per_experiment = []
ts_margin_agg_per_experiment = []

ucb_reward_agg_per_experiment = []
ucb_margin_agg_per_experiment = []

gr_reward_per_experiment = []
gr_margin_per_experiment = []

ts_reward_per_day = []
ts_margin_per_day = []

ucb_reward_per_day = []
ucb_margin_per_day = []

gr_reward_per_day = []
gr_margin_per_day = []

ts_reward_agg_per_day = []
ts_margin_agg_per_day = []

ucb_reward_agg_per_day = []
ucb_margin_agg_per_day = []

products_margin = [prod1_margin1, prod2_margin2, prod3_margin3, prod4_margin4, prod5_margin5]
products_conversion = [conv_rate1, conv_rate2, conv_rate3, conv_rate4, conv_rate5]
products_prices = [prod1_price1, prod2_price2, prod3_price3, prod4_price4, prod5_price5]
# products_weights = [P1_weights, P2_weights, P3_weights, P4_weights, P5_weights]
products_customers = [n_customers1, n_customers2, n_customers3, n_customers4, n_customers5]

# aggregated values
products_margin_agg = [sum(sub_list) / len(sub_list) for sub_list in products_margin]
products_conv_agg = [sum(sub_list) / len(sub_list) for sub_list in products_conversion]
# products_prices_agg = [sum(sub_list) / len(sub_list) for sub_list in zip(*products_prices)]
products_cust_agg = ([sum(sub_list) / len(sub_list) for sub_list in products_customers])

# optimum
opt = []
for i in range(n_prods):
    opt.append(np.max(products_conversion[i]))

#################### Execution ####################
for i in range(n_experiments):
    for t in range(T):
        ts_learner = TS_Learner(n_arms)
        ucb_learner = UCB_Learner(n_arms)
        gr_learner = Greedy_Learner(n_arms)

        ts_learner_agg = TS_Learner(n_arms)
        ucb_learner_agg = UCB_Learner(n_arms)
        gr_learner_agg = Greedy_Learner(n_arms)
        for pid in range(n_prods):
            visited_products = [pid]
            env_agg = Environment(n_arms, products_cust_agg[pid], products_margin_agg,
                                  products_conv_agg[pid])

            for j in range(n_prods):
                env = Environment(n_arms, products_customers[j], products_margin[j],
                                  products_conversion[j])
                weight = adjacent_matrix[pid][j]
                # Thompson Sampling Learner
                # condition to skip product if it was visited or has no weight
                if j not in visited_products and weight > 0.0:
                    visited_products.append(j)
                # Thompson Sampling Learner
                    ts_arm_pulled = ts_learner.pull_arm()
                    ts_margin = env.margin_mode(ts_arm_pulled, weight)
                    ts_reward = env.reward_mode(ts_arm_pulled, weight)
                    ts_learner.update(ts_arm_pulled, ts_reward, ts_margin)

                # UCB1 Learner
                    ucb_arm_pulled = ucb_learner.pull_arm()
                    ucb_margin = env.margin_mode(ucb_arm_pulled, weight)
                    ucb_reward = env.reward_mode(ucb_arm_pulled, weight)
                    ucb_learner.update(ucb_arm_pulled, ucb_reward, ucb_margin)

                # Greedy Learner
                    gr_arm_pulled = gr_learner.pull_arm()
                    gr_margin = env.margin_mode(gr_arm_pulled, weight)
                    gr_reward = env.reward_mode(gr_arm_pulled, weight)
                    gr_learner.update(gr_arm_pulled, gr_reward, gr_margin)

                # aggregate environment-learners
                    # TS learner
                    ts_agg_arm_pulled = ts_learner_agg.pull_arm()
                    ts_agg_margin = env_agg.margin_agg_mode(ts_agg_arm_pulled, weight)
                    ts_agg_reward = env_agg.reward_agg_mode(ts_agg_arm_pulled, weight)
                    ts_learner_agg.update(ts_agg_arm_pulled, ts_agg_reward, ts_agg_margin)

                    # UCB1 Learner
                    ucb_agg_arm_pulled = ucb_learner_agg.pull_arm()
                    ucb_agg_margin = env_agg.margin_agg_mode(ucb_agg_arm_pulled, weight)
                    ucb_agg_reward = env_agg.reward_agg_mode(ucb_agg_arm_pulled, weight)
                    ucb_learner_agg.update(ucb_agg_arm_pulled, ucb_agg_reward, ucb_agg_margin)

            visited_products = []

        ts_reward_agg_per_day.append(ts_learner.collected_rewards)
        ucb_reward_agg_per_day.append(ucb_learner.collected_rewards)
        ts_margin_agg_per_day.append(ts_learner.collected_margins)

        ts_reward_per_day.append(ts_learner.collected_rewards)
        ucb_reward_per_day.append(ucb_learner.collected_rewards)
        ts_margin_per_day.append(ts_learner.collected_margins)
        gr_margin_per_day.append(gr_learner.collected_margins)

    # for disaggregate algorithm performance (Bandit comparison)
    ts_reward_per_experiment.append(ts_reward_per_day)
    ucb_reward_per_experiment.append(ucb_reward_per_day)

    # for disaggregate margin performance
    ts_margin_per_experiment.append(ts_margin_per_day)
    gr_margin_per_experiment.append(gr_margin_per_day)

    # for aggregate algorithm performance (Bandit comparison)
    ts_reward_agg_per_experiment.append(ts_reward_per_day)
    ucb_reward_agg_per_experiment.append(ucb_reward_per_day)

    # for aggregate margin performance
    ts_margin_agg_per_experiment.append(ts_margin_per_day)

ts_margin_per_experiment_transposed = list(map(list, zip(*ts_margin_per_experiment)))
gr_margin_per_experiment_transposed = list(map(list, zip(*gr_margin_per_experiment)))
ts_reward_per_experiment_transposed = list(map(list, zip(*ts_reward_per_experiment)))
ucb_reward_per_experiment_transposed = list(map(list, zip(*ucb_reward_per_experiment)))

ts_margin_agg_per_experiment_transposed = list(map(list, zip(*ts_margin_agg_per_experiment)))
ts_reward_agg_per_experiment_transposed = list(map(list, zip(*ts_reward_agg_per_experiment)))
ucb_reward_agg_per_experiment_transposed = list(map(list, zip(*ucb_reward_agg_per_experiment)))


#################### Algorithm Performance per day (Regret) ####################
plt.figure(figsize=(8, 6), dpi=80)
plt.title('Regret over time')
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(np.max(opt) - ts_reward_per_experiment_transposed, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(np.max(opt) - ucb_reward_per_experiment_transposed, axis=0)), 'b')
plt.legend(['Thompson Sampling', 'UCB1'], loc='lower right')
plt.show()

#################### Margin Performance ####################
print('MARGIN Disaggregate PERFORMANCE')
print()
print("Thompson Sampling Algorithm")
for i in range(n_prods):
    print(
        "Selected price for product " + str(i + 1) + " :  [{}]".format(
            products_prices[i][np.argmax([len(a) for a in ts_learner.rewards_per_arm])]))
print('Total margin:  [{}]'.format(np.sum(ts_margin_per_experiment_transposed)))
print()
print("Greedy Algorithm")
for i in range(n_prods):
    print(
        "Selected price for product " + str(i + 1) + " :  [{}]".format(
            products_prices[i][np.argmax([len(a) for a in gr_learner.rewards_per_arm])]))
print('Total margin:  [{}]'.format(np.sum(gr_margin_per_experiment_transposed)))
print()

plt.figure(figsize=(8, 6), dpi=80)
plt.title('Cumulative Margin')
plt.xlabel("Days")
plt.ylabel("Margin")
plt.plot(np.cumsum(ts_margin_per_experiment_transposed), 'g', label='Thompson Sampling')
plt.plot(np.cumsum(gr_margin_per_experiment_transposed), 'r', label='Greedy')
plt.legend(['Thompson Sampling', 'Greedy'], loc='lower right')
plt.show()


#################### Algorithm Performance per product (Regret) ####################
# for i in range(n_prods):
#     plt.figure(figsize=(8, 6), dpi=80)
#     plt.title('Regret over time - Product ' + str(i + 1))
#     plt.xlabel("t")
#     plt.ylabel("Regret")
#     plt.plot(np.cumsum(np.mean(opt[i] - ts_reward_per_experiment_transposed[i], axis=0)), 'g')
#     plt.plot(np.cumsum(np.mean(opt[i] - ucb_reward_per_experiment_transposed[i], axis=0)), 'b')
#     plt.legend(['Thompson Sampling', 'UCB1'], loc='lower right')
#     plt.show()
#
#     #################### Margin Performance ####################
#     print('MARGIN PERFORMANCE')
#     print()
#     print("Thompson Sampling Algorithm")
#     print(
#         "Selected price for product " + str(i + 1) + " :  [{}]".format(
#             products_prices[i][np.argmax([len(a) for a in ts_learner.rewards_per_arm])]))
#     print('Total margin:  [{}]'.format(np.sum(ts_margin_per_experiment_transposed[i])))
#     print()
#     print("Greedy Algorithm")
#     print(
#         "Selected price for product " + str(i + 1) + " :  [{}]".format(
#             products_prices[i][np.argmax([len(a) for a in gr_learner.rewards_per_arm])]))
#     print('Total margin:  [{}]'.format(np.sum(gr_margin_per_experiment_transposed[i])))
#     print()
#
#     plt.figure(figsize=(8, 6), dpi=80)
#     plt.title('Cumulative Margin')
#     plt.xlabel("Days")
#     plt.ylabel("Margin")
#     plt.plot(np.cumsum(ts_margin_per_experiment_transposed[i]), 'g', label='Thompson Sampling')
#     plt.plot(np.cumsum(gr_margin_per_experiment_transposed[i]), 'r', label='Greedy')
#     plt.legend(['Thompson Sampling', 'Greedy'], loc='lower right')
#     plt.show()
