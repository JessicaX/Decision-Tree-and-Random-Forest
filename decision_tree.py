from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        # self.tree = []
        self.tree = {}
        self.removed_attr = []

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        max_info_gain = 0
        max_attr, max_val = 0, 0
        if len(y) == 0: # no data to split
            return None

        if len(self.removed_attr) == len(X[0]): #all attributes splited 
            return {"result": max(y, key=y.count), "left": None, "right": None, "type": "leaf"}
        
        if entropy(y) == 0: #all contains same label
            return {"result": y[0], "left": None, "right": None, "type": "leaf"} #if entropy is 0 means all y are the same return label value

        for col in range(len(X[0])):
            if col in self.removed_attr:
                continue
            for row in range(len(X)):
                X_left, X_right, y_left, y_right = partition_classes(X, y, col, X[row][col])
                info_gain = information_gain(y, [y_left, y_right])
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    max_attr = col
                    max_val = X[row][col]

        X_left, X_right, y_left, y_right = partition_classes(X, y, max_attr, max_val)
        node = {"value": max_val, "attr": max_attr, "left": {}, "right": {}}
        self.removed_attr.append(max_attr)

        if len(self.tree) == 0:
            self.tree = node

        left = self.learn(X_left, y_left)
        right = self.learn(X_right, y_right)
        node["left"] = left
        node["right"] = right
        if left is None: #if left or right is None, means no right to split, should return parent's mode value
            node["left"] = {"result": max(y, key=y.count), "left": None, "right": None, "type": "leaf"}

        if right is None:
            node["right"] = {"result": max(y, key=y.count), "left": None, "right": None, "type": "leaf"}

        node["type"] = "branch"
        return node

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        node = self.tree

        while node["type"] != "leaf":
            val = record[node["attr"]]
            if type(val) is str:
                if val == node["value"]:
                    node = node["left"]
                else:
                    node = node["right"]
            else:
                if val <= node["value"]:
                    node = node["left"]
                else:
                    node = node["right"]
                
        return node["result"]

# if __name__ == "__main__":
#     X = [[3, 'aa', 10], [1, 'bb', 22], [2, 'cc', 28], [5, 'bb', 32], [4, 'cc', 32]]  
#     y = [1, 1, 0, 0, 1]
#     d = DecisionTree()
#     print(d.learn(X, y))
#     print(d.tree)
#     print(d.classify([4, 'cc', 32]))