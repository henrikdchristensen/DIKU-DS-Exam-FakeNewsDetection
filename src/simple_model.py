from sklearn.dummy import DummyClassifier
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional
from pipeline import FunctionApplier
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from pipeline import apply_pipeline


#reliable: reliable, clickbait, political
#fake: fake, conspiracy, junksci, hate, unreliable, bias, satire, (state)
#False for fake, True for reliable
binary_labels: dict = {
    'fake':False,
    'conspiracy':False,
    'junksci':False,
    'hate':False,
    'unreliable':False,
    'bias':False,
    'satire':False,
    'state':False,
    'reliable':True,
    'clickbait':True,
    'political':True,
    'nan': False
}

#Models based on content
def content_model(train_data, train_labels, val_data, val_labels, strategy : str):
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(train_data, train_labels)
    pred = dummy_clf.predict(val_data)
    acc = dummy_clf.score(val_labels, pred)
    return acc

#Accuracy of three different models - iterate for each chunk
#freq_model = content_model(data, strategy="most_frequent")
#uniform_model = content_model(data, strategy="uniform")
#print(f"Most frequent acc: {freq_model}"
#print(f"Uniform acc: {uniform_model}")


# def sourceDistribution(df: pd.DataFrame):

#     keys = df['type'].unique()
#     print(keys)

#     # make dict of sources and fake label count
#     sourceDict = {}
#     typeDict = {k: 0 for k in keys}

#     for index, row in df.iterrows():
#         print(row['domain'])
#         typeDict[row['type']] += 1
#         if not row['domain'] in sourceDict:
#             sourceDict[row['domain']] = {k: 0 for k in keys}
#             sourceDict[row['domain']]['total'] = 0

#         # src = sourceDict[row['domain']]
#         sourceDict[row['domain']][row['type']] += 1
#         sourceDict[row['domain']]['total'] += 1

#     return sourceDict

# def sourceDistribution(df: pd.DataFrame):

#     keys = df['type'].unique()
#     print(keys)

#     # make dict of sources and fake label count
#     sourceDict = {}
#     typeDict = {k: 0 for k in keys}

#     for index, row in df.iterrows():
#         print(row['domain'])
#         typeDict[row['type']] += 1
#         if not row['domain'] in sourceDict:
#             sourceDict[row['domain']] = {k: 0 for k in keys}
#             sourceDict[row['domain']]['total'] = 0

#         # src = sourceDict[row['domain']]
#         sourceDict[row['domain']][row['type']] += 1
#         sourceDict[row['domain']]['total'] += 1

#     return sourceDict

# # Dict[str, int] = {"A": 1} # Mapping of str to int

# def simpleMajorityClassfier(sourceDict: Dict[str, Dict[str, int]]):

#     # count all the fake labels for each source, if the count is greater than half of the total, then it is fake
#     for source in sourceDict:
#         total = sourceDict[source]['total']
#         count = 0
#         for label in sourceDict[source]:
#             if label != 'total':
#                 count += sourceDict[source][label]

#         if count > total/2:
#             sourceDict[source]['fake'] = 1
#         else:
#             sourceDict[source]['fake'] = 0
#     pass
    

class Simple_model(FunctionApplier):

    def __init__(self, keys):
        self.dict_domains = {}
        self.sourceDict = {}
        self.typeDict = {k: 0 for k in binary_labels}
        # make result new dataframe
        self.result = pd.DataFrame(columns=['domain', 'fake'])

    def sourceDistribution(self, row):
        print(row['domain'])
        
        if row['type'] in binary_labels:
            self.typeDict[row['type']] += 1
            
        if not row['domain'] in self.sourceDict:
            self.sourceDict[row['domain']] = {k: 0 for k in binary_labels}
            self.sourceDict[row['domain']]['total'] = 0
        # src = sourceDict[row['domain']]

        if row['type'] in binary_labels:
            self.sourceDict[row['domain']][row['type']] += 1
            self.sourceDict[row['domain']]['total'] += 1

        # return sourceDict
    
    def simpleMajorityClassfier( self, sourceDict: Dict[str, Dict[str, int]]):
        # count all the fake labels for each source, if the count is greater than half of the total, then it is fake
        # make dict of sources and fake label count as a pandas dataframe
        print("sourceDict: ", sourceDict)
        for source in sourceDict:
            total = sourceDict[source]['total'] | 0
            count = 0
            for label in sourceDict[source]:
                if label != 'total':
                    # only count the labels that are in the binary_labels dict
                    if label in binary_labels:
                        # only add to count if the label is fake
                        if binary_labels[label] == False:
                            count += sourceDict[source][label]

            if count > total/2:
                sourceDict[source]['isTrusty'] = 1
            else:
                sourceDict[source]['isTrusty'] = 0
            # calculate the percentage of fake labels
            if total != 0:
                sourceDict[source]['percentFake'] = count/total * 100

        panda = pd.DataFrame.from_dict(sourceDict, orient='index')
        self.result = panda


    def function_to_apply(self, row: pd.DataFrame):
        # ASKE DO YOUR THING
        self.sourceDistribution(row)
        # self.simpleMajorityClassfier(self.sourceDict)
        return row
    

class Simple_logistic_model(FunctionApplier):

    def __init__(self, keys):
        # make result new dataframe
        self.result = pd.DataFrame(columns=['domain', 'fake'])

    def sourceDistribution(self):
        df = pd.read_csv('../datasets/sample/news_sample.csv')

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df['content'], df['type'], test_size=0.2, random_state=42)

        # Convert the text into a matrix of token counts
        vectorizer = CountVectorizer(stop_words='english')
        X_train_counts = vectorizer.fit_transform(X_train)
        X_test_counts = vectorizer.transform(X_test)

        # Train a logistic regression model
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_counts, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test_counts)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='fake')
    
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 score (fake class): {f1:.2f}")




def simple_model_test():
    sm = Simple_model(binary_labels)
    apply_pipeline("../datasets/sample/news_sample.csv", [
        sm
    ])

    # print("got source dict", sm.sourceDict)
    sm.simpleMajorityClassfier(sm.sourceDict)

    print("got result", sm.result)



def simple_model_test2():
    sm = Simple_logistic_model(binary_labels)
    sm.sourceDistribution()
    # print("got source dict", sm.sourceDict)

# simple_model_test2()