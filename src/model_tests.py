import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, vstack, load_npz, save_npz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time


def split_data(data, features, y, set="set", get_val=True):
    train = data[data[set] == 0]
    val = data[data[set] == 1]
    test = data[data[set] == 2]
    X_train, y_train = train[features], (train[y].astype(int))
    X_val, y_val = val[features], (val[y].astype(int))
    X_test, y_test = test[features], (test[y].astype(int))
    if not get_val:
        return X_train, X_test, y_train, y_test
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_count_vector(X_train, X_val, X_test):
    # count vectri
    count_vectorizer = CountVectorizer(ngram_range=(1, 1)) # unigram

    # fit and transform train data to count vectorizer
    count_vectorizer.fit(X_train)

    return (count_vectorizer.transform(X_train),
            count_vectorizer.transform(X_val),
            count_vectorizer.transform(X_test), count_vectorizer)


def create_tdfidf_vector(X_train, X_val, X_test, ngram_range=(1, 1)):
    # tfidf vector
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range) # unigram

    # fit and transform train data to tfidf vectorizer
    tfidf_vectorizer.fit(X_train)

    return (tfidf_vectorizer.transform(X_train),
            tfidf_vectorizer.transform(X_val),
            tfidf_vectorizer.transform(X_test), tfidf_vectorizer)


def create_tdfidf_vector_unigram(X_train, X_val, X_test):
    return create_tdfidf_vector(X_train, X_val, X_test, ngram_range=(1, 1))

def create_tdfidf_vector_bigram(X_train, X_val, X_test):
    return create_tdfidf_vector(X_train, X_val, X_test, ngram_range=(1, 2))

def create_tdfidf_vector_trigram(X_train, X_val, X_test):
    return create_tdfidf_vector(X_train, X_val, X_test, ngram_range=(1, 3))

def save_csr_picle(file, vectors, append=False):
    file_param = "wb" if not append else "ab"
    with open(file, file_param) as f:
        for train, val, test in vectors:
            pickle.dump(train, f)
            pickle.dump(val, f)
            pickle.dump(test, f)


def apply_vec_func(file, func, name, X_train, X_val, X_test):
    start_time = time()
    X_train_vec, X_val_vec, X_test_vec, vectorizer = func(X_train, X_val, X_test)
    save_csr_picle(file, [(X_train_vec, X_val_vec, X_test_vec)], append=True)
    print(f"Saved {name} vectors in {time() - start_time} seconds") 
    return vectorizer

# Vectorize data
def create_vector_file(file, vec_funcs, X_train, X_val, X_test, y_train, y_val, y_test, save_y=True, append_y=False):
    # save y
    vectorizers = []
    if save_y:
        save_csr_picle(file, [(y_train, y_val, y_test)], append=append_y)
    # vectorize X and save
    for func, name in vec_funcs:
        vectorizers.append(apply_vec_func(file, func, name, X_train, X_val, X_test))
    return vectorizers

def create_vector_file_cols(file, x_cols, y_col, vec_funcs, out_file, set_col = "set"):
    data = pd.read_csv(file, usecols=x_cols + [y_col, set_col])
    print("csv read for file:", file)

    for i, col in enumerate(x_cols):
        print("Vectorizing column:", col)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, col, y_col, set=set_col)
        save_y = True if i == 0 else False
        create_vector_file(out_file, vec_funcs, X_train, X_val, X_test, y_train, y_val, y_test, save_y=save_y)

def create_vectors_from_infolist(out_file, info_list, use_standard=True):
    vectorizers = []
    for i, item in enumerate(info_list):
        if use_standard:
            file, x_col, vec_func, model, test_name = item
            y_col = "type_binary"
            set_col = "set"
        else:
            file, x_col, y_col, set_col, vec_func, model, test_name = item
        start_time = time()
        data = pd.read_csv(file, usecols=[x_col, y_col, set_col])
        print(f"Creating vector: {test_name} (data read in {time() - start_time} seconds)")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, x_col, y_col, set=set_col)
        append_y = False if i == 0 else True
        vectorizers.append(create_vector_file(out_file, [(vec_func, test_name)], X_train, X_val, X_test, y_train, y_val, y_test, save_y=True, append_y=append_y)[0])
    return vectorizers

def try_models(models, X_train, X_val, y_train, y_val, name=None, predict_only=False):
    metrics = []
    for model in models:
        start_time = time() 
        model.fit(X_train, y_train)
        train_time = time() - start_time
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_val)
        
        if name == None:
            name = type(model).__name__
        metrics.append({
            "name": name,
            "train_acc": accuracy_score(y_train, y_train_pred),
            "val_acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred), 
            "time": "{:.2f}".format(train_time),
            'model': model
        })
        print(f"{name} finished in {(time() - start_time):.2f} seconds")
    return pd.DataFrame(metrics)

def get_predict_metrics(model, X_test, y_test, name=None):
    metrics = []
    y_pred = model.predict(X_test)
    
    if name == None:
        name = type(model).__name__
    metrics = [({
        "name": name,
        "acc": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred), 
        'model': model
    })]

    return pd.DataFrame(metrics)


class Test_statistic():
    def __init__(self):
        self.metrics = pd.DataFrame()

    def test_baseline(self, X_train, X_val, y_train, y_val, model, name=None):
        metric = try_models([model], X_train, X_val, y_train, y_val, name=name)
        self.metrics = pd.concat([self.metrics, metric])
    
    def add_vectorizer_col(self, vectorizers):
        self.metrics["vectorizer"] = vectorizers


def test_vectors(models, vec_funcs, file, tests = None):
    if tests == None:
        tests = Test_statistic()
    with open(file, 'rb') as f:
        y_train, y_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))
        for _, name in vec_funcs:
            X_train, X_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))
            for model in models:
                tests.test_baseline(X_train, X_val, y_train, y_val, model, name=name)
    return tests

def test_vectors_cols(file, models, vec_funcs_used, cols_in_file, test_col = None, test_vec = None, tests = None):
    if tests == None:
        tests = Test_statistic()
    with open(file, 'rb') as f:
        y_train, y_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))
        for col in cols_in_file:
            for _, name in vec_funcs_used:
                X_train, X_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))

                if test_col != None and col != test_col:
                    # skip if not test column
                    continue
                if test_vec != None and name != test_vec:
                    # skip if not test vector
                    continue
                if test_col != None and test_vec != None:
                    print("Testing")

                for model in models:
                    if type(model) == tuple:
                        model, model_name = model
                        col = f"{col}_{model_name}"
                    tests.test_baseline(X_train, X_val, y_train, y_val, model, name=f"{col}_{name}")
    return tests

def test_vectors_from_infolist(from_file, info_list, tests = None, use_standard=True):
    if tests == None:
        tests = Test_statistic()
    with open(from_file, 'rb') as f:
        for i, item in enumerate(info_list):
            if use_standard:
                file, x_col, vec_func, model, test_name = item
                y_col = "type_binary"
                set_col = "set"
            else:
                file, x_col, y_col, set_col, vec_func, model, test_name = item
            y_train, y_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))
            X_train, X_val, _ = (pickle.load(f), pickle.load(f), pickle.load(f))
            tests.test_baseline(X_train, X_val, y_train, y_val, model, name=test_name)
    return tests