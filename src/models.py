
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import pipeline as pp
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import transformers as ppb # pytorch-pretrained-bert
import torch

# The number of rows to train the model
BATCH_SIZE = 1000000

def Vectorize(to_csv=False):
    vectorized_data = {}
    files = [("train", "../datasets/sample/train.csv"),
             ("test", "../datasets/sample/test.csv"),
             ("val", "../datasets/sample/vali.csv")]
    
    vect = None
    for label, file in files:
        cleaned_data = pp.apply_pipeline(file, [
            (pp.binary_labels(), 'type'),
            (pp.Clean_data(), 'content')
        ], 
        get_batch=True, 
        batch_size=BATCH_SIZE)

        #TODO: how to avoid leaking information from the test set? 
        if vect is None:
            vect = TfidfVectorizer()
            vect.fit(cleaned_data['content'])
        
        content_tfidf = vect.transform(cleaned_data['content'])
        
        vectorized_data["X_" + label] = content_tfidf
        vectorized_data["y_" + label] = cleaned_data['type']

        if to_csv:
            content_tfidf_df = pd.DataFrame(content_tfidf.todense(),columns = vect.get_feature_names_out())
            content_tfidf_df.to_csv("../datasets/sample/" + label + "_tfidf.csv", index=False)

    return vectorized_data

def k_neighbors_classifier(data):
    # Define the classifier classes
    k_nearest = KNeighborsClassifier(n_neighbors=15, weights='distance')

    # Fit the model
    k_nearest.fit(data["X_train"], data["y_train"].astype(int))

    # Predict on the test set
    k_nearest_pred = k_nearest.predict(data["X_test"])

    # Evaluate performance
    print("k_nearest accuracy:", accuracy_score(data["y_test"].astype(int), k_nearest_pred))


def support_vector_classifier(data):
    # Define the classifier classes
    svc = SVC(kernel='linear')

    # Fit the model
    svc.fit(data["X_train"], data["y_train"].astype(int))

    # Predict on the test set
    svc_pred = svc.predict(data["X_test"])

    # Evaluate performance
    print("svc accuracy:" + str(accuracy_score(data["y_test"].astype(int), svc_pred)))


def distilBERT():
    content = pp.apply_pipeline("../datasets/sample/train.csv", [], get_batch=True, batch_size=BATCH_SIZE)['content']

    #Tokenizer and model input
    pretrained_weights = 'distilbert-base-uncased'
    tokenizer = ppb.DistilBertTokenizer.from_pretrained(pretrained_weights)
    model = ppb.DistilBertModel.from_pretrained(pretrained_weights, from_tf=True)

    #Tokenize input
    tokenized = content.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    model.eval()

    #Pad input so that all sequences are of the same size:
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    padded = padded[:,:32]

    # Tell embedding model to disregard pad tokens
    attention_mask = np.where(padded != 0, 1, 0)
    
    device = torch.device("cpu")

    if torch.cuda.is_available():
      model = model.cuda()
      device = torch.device("cuda")

    # Convert input to a pytorch tensor
    input = torch.tensor(np.array(padded), device=device)
    attention_mask = torch.tensor(attention_mask, device=device)

    # Embed sequences (processing in batches to avoid memory problems)
    batch_size= 200
    embeddings = []

    for start_index in range(0, input.shape[0], batch_size):
      with torch.no_grad():
        # Call embedding model
        embedding = model(input[start_index:start_index+batch_size], 
                          attention_mask=attention_mask[start_index:start_index+batch_size])[0][:,0,:]
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings)   # concatenate all batch outputs back into one tensor

    # Move embeddings back to numpy
    embeddings = embeddings.cpu().numpy()
    
    return embeddings

def word_embeddings_model(word_embeddings):
    print("a")
    labels = pp.apply_pipeline("../datasets/sample/train.csv", [], get_batch=True, batch_size=BATCH_SIZE)['type']
    X_train, X_test, y_train, y_test = train_test_split(word_embeddings, labels, test_size=0.2, random_state=42)
    print("b")
    layers = [1,2,3,4,5]
    layer_sizes = [2,5,8,11,14]
    tuple_list = []

    for layer_size in layer_sizes:
        for layer in layers:
            tuple_list.append((layer_size,)*layer)
            
    inputs = {'hidden_layer_sizes': tuple_list}

    # Define the classifier classes
    MLP = MLPClassifier()

    print("c")
    #Gridsearch
    cross_val = GridSearchCV(MLP, inputs)
    print("d")
    # Fit the model
    cross_val.fit(X_train, y_train)


#vectorized_data = Vectorize()
#k_neighbors_classifier(vectorized_data)
#support_vector_classifier(vectorized_data)

