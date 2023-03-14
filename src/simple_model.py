from sklearn.dummy import DummyClassifier
from typing import List, Dict, Union, Tuple, Iterator, Callable, Optional
from pipeline import FunctionApplier
import pandas as pd


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
    'political':True
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

def sourceDistribution(df: pd.DataFrame):

    keys = df['type'].unique()
    print(keys)

    # make dict of sources and fake label count
    sourceDict = {}
    typeDict = {k: 0 for k in keys}

    for index, row in df.iterrows():
        print(row['domain'])
        typeDict[row['type']] += 1
        if not row['domain'] in sourceDict:
            sourceDict[row['domain']] = {k: 0 for k in keys}
            sourceDict[row['domain']]['total'] = 0

        # src = sourceDict[row['domain']]
        sourceDict[row['domain']][row['type']] += 1
        sourceDict[row['domain']]['total'] += 1

    return sourceDict

# Dict[str, int] = {"A": 1} # Mapping of str to int

def simpleMajorityClassfier(sourceDict: Dict[str, Dict[str, int]]):

    # count all the fake labels for each source, if the count is greater than half of the total, then it is fake
    for source in sourceDict:
        total = sourceDict[source]['total']
        count = 0
        for label in sourceDict[source]:
            if label != 'total':
                count += sourceDict[source][label]

        if count > total/2:
            sourceDict[source]['fake'] = 1
        else:
            sourceDict[source]['fake'] = 0
    pass
    

class Simple_model(FunctionApplier):

    def __init__(self, keys):
        self.dict_domains = {}
        sourceDict = {}
        typeDict = {k: 0 for k in keys}

    def sourceDistribution(self, df: pd.DataFrame):
        keys = df['type'].unique()
        print(keys)
        # make dict of sources and fake label count
        self.sourceDict = {}
        self.typeDict = {k: 0 for k in keys}

        for index, row in df.iterrows():
            print(row['domain'])
            self.typeDict[row['type']] += 1
            if not row['domain'] in self.sourceDict:
                self.sourceDict[row['domain']] = {k: 0 for k in keys}
                self.sourceDict[row['domain']]['total'] = 0
            # src = sourceDict[row['domain']]
            self.sourceDict[row['domain']][row['type']] += 1
            self.sourceDict[row['domain']]['total'] += 1

        # return sourceDict
    
    def simpleMajorityClassfier(sourceDict: Dict[str, Dict[str, int]]):
        # count all the fake labels for each source, if the count is greater than half of the total, then it is fake
        # make dict of sources and fake label count as a pandas dataframe
        for source in sourceDict:
            total = sourceDict[source]['total']
            count = 0
            for label in sourceDict[source]:
                if label != 'total':

                    # only count the labels that are in the binary_labels dict
                    # if label in binary_labels:
                        # only add to count if the label is fake
                    if binary_labels[label] == False:
                        count += sourceDict[source][label]

            if count > total/2:
                sourceDict[source]['fake'] = 1
            else:
                sourceDict[source]['fake'] = 0

        panda = pd.DataFrame.from_dict(sourceDict, orient='index')        

        return panda


    def function_to_apply(self, row):
        # ASKE DO YOUR THING
        





        

        return row