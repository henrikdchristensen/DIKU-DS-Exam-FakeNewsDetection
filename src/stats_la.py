from hashlib import new
from xmlrpc.client import Binary
from pytz import country_names
import pipeline as pp
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
import pipeline as pp

class counter(pp.FunctionApplier):
    def __init__(self, to_count : str):
        self.to_count = to_count

    def function_to_apply(self, row):
        return row['content'].count(self.to_count)

class word_count(pp.FunctionApplier):
    def function_to_apply(self, row):
        return len(row['content'])

def boxplot(col_true, col_false, title : str):    
    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    # Pandas dataframe
    data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

    # Plot the dataframe
    ax = data[['Reliable', 'Fake']].plot(kind='box', title=title)

    # Display the plot
    plt.show()

def barplot(col_true, col_false, title : str):
    len_true = len(col_true)
    len_false = len(col_false)
    true_height = sum(col_true)/len_true
    false_height = sum(col_false)/len_false
    data = pd.DataFrame({"Reliable": true_height, "Fake": false_height})

    height = [true_height, false_height]
    colors = ['green', 'purple']


    plt.bar(height=data, tick_label=['true', 'false'], color=colors)

    plt.ylabel("Number of occurrences")
    plt.xlabel("Label")
    plt.show()
