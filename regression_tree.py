# Practice example for Regression Trees using MSE as split criteria
# This is not optimised (esp. not performance wise), just my first best implementation of a regression tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class RegressionTree():

    class node():
        def __init__(self, split_value=None, split_feature=None, sub_nodes=(None, None)):
            self.split_value=split_value
            self.split_feature=split_feature
            self.sub_nodes=sub_nodes
        
        def append_node(self, parent_node, split_value, direction):
            parent_node.sub_nodes[direction] = RegressionTree.node(split_value=split_value)

    def __init__(self):
        self.root = RegressionTree.node()

    def fit(self, X, min_points, max_depth):

        def get_best_split(x):
            error = []

            # get splits with minimum MSE per feature
            for feature in range(x.shape[1]-1):
                x_sorted = x[x[:, feature].argsort()]
                mse = {}

                # get MSE values for all possible splits within the feature
                for split in range(1, x.shape[0]):
                    left_data, right_data = x_sorted[:split], x_sorted[split:]
                    left_mean, right_mean = left_data[:,-1].mean(), right_data[:, -1].mean()
                    mse[x_sorted[split-1,feature]] = mean_squared_error(left_mean*np.ones(left_data.shape[0]),left_data[:,-1]) + mean_squared_error(right_mean*np.ones(right_data.shape[0]), right_data[:,-1])

                # add split with that features minimum MSE
                error.append((list(mse.keys())[list(mse.values()).index(min(mse.values()))], min(mse.values())))

            # minimum MSE split will be used as node
            error_values = [err for _, err  in error]
            split_feature = error_values.index(min(error_values))
            split_value = [val for val, _  in error][split_feature]
            return split_feature, split_value
        
        def create_split(x, node, depth):
            if x.shape[0] >= min_points and depth <= max_depth:
                node.split_feature, node.split_value = get_best_split(x)
                node.sub_nodes = (RegressionTree.node(), RegressionTree.node())
                left_data, right_data = x[x[:, node.split_feature] <= node.split_value], x[x[:, node.split_feature] > node.split_value]
                create_split(left_data, node.sub_nodes[0], depth+1)
                create_split(right_data, node.sub_nodes[1], depth+1)
            else:
                node.split_value = x[:,-1].mean()

        create_split(X, self.root, depth=1)

    def predict(self, y):
        current_node = self.root
        while current_node.sub_nodes != (None, None):
            if y[current_node.split_feature] <= current_node.split_value:
                current_node = current_node.sub_nodes[0]
            else:
                current_node = current_node.sub_nodes[1]
        return current_node.split_value

    def print_tree(self):
        
        def print_node(node):
            print(f"F:{node.split_feature}, V:{node.split_value}")
            if node.sub_nodes != (None, None):
                print_node(node.sub_nodes[0])
                print_node(node.sub_nodes[1])
        
        print_node(self.root)

data = np.array([[1., 0.1], [4.8, 1.1], [6.1, 1.4], [5.9, 1.2], [2.1, 0.15], [1.7, 0.2], [1.2, 0.1], [2.2, 0.2], [4.3, 0.1], [2.8, 0.1], [3.2, 0.15], [4.2, 1.1], [5.3, 1.]])

model = RegressionTree()
model.fit(data, min_points=5, max_depth=1)

x = np.array([4])
prediction = model.predict(x)
print(f"Prediction for x = [{4}]: {round(prediction, 2)}")

plt.scatter(data[:, 0], data[:, 1])
plt.scatter(x, prediction)