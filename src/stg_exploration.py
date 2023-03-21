from hashlib import new
from xmlrpc.client import Binary
from pytz import country_names
import pipeline as pp
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.stem import PorterStemmer
import pipeline as pp


#Count the number of times the <NUM> label occurs in content. Requires clean_data not to clean <NUM>
class count_number_occurences_per_label(pp.FunctionApplier):
    def __init__(self):
        keys = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 'satire', 'state', 
                'reliable', 'clickbait', 'political', 'rumor']
        self.typeDict = {k: 0 for k in keys}
        self.countDict = {k: 0 for k in keys}
        self.perTypeDict = {}
        self.binaryDict = {'Reliable': 0, 'Fake': 0}
        self.count_df = pd.DataFrame(columns=['id', 'label', 'length'])
        self.fake = []
        self.reliable = []
        self.lookUpDict = pp.Binary_labels()

    #def as_panda(df):
        #df = read_csv()


    def function_to_apply2(self, row):

        # if(row["type"])

        try:
            self.countDict[row['type']] += row['content'].count("<number>")
            self.typeDict[row['type']] += 1
        except:
            pass

        for k, v in self.countDict.items():
            try:
                self.perTypeDict[k] = v/self.typeDict[k]
            except:
                self.perTypeDict[k] = 0
        return row['content'].count("<number>")



    def function_to_apply(self, row):
        try:
            self.countDict[row['type']] += row['content'].count("<number>")
            self.typeDict[row['type']] += 1
        except:
            pass

        for k, v in self.countDict.items():
            try:
                self.perTypeDict[k] = v/self.typeDict[k]
            except:
                self.perTypeDict[k] = 0
        return row['content'].count("<number>")
        #return "test"

    def print_stats(self):
        print(self.perTypeDict)
        
    def plot_stats(self):
        types = list(self.perTypeDict.keys())
        counts = list(self.perTypeDict.values())
        colors = ['purple', 'purple', 'purple', 'purple', 'green', 'purple', 'purple', 'green', 'purple', 'green', 'purple']
        plt.bar(range(len(self.perTypeDict)), counts, tick_label=types, color=colors)
        plt.xlabel("Number of occurrence")
        plt.ylabel("Label")
        plt.show()
    
    def plot_stats_binary(self):
        true_types = ['reliable', 'political', 'clickbait']
        for type in list(self.perTypeDict.keys()):
            if type in true_types:
                self.binaryDict['Reliable'] += self.perTypeDict[type]
            else:
                self.binaryDict['Fake'] += self.perTypeDict[type]
        types = list(self.binaryDict.keys())
        counts = list(self.binaryDict.values())
        colors = ['green', 'purple']
        plt.bar(range(len(self.binaryDict)), counts, tick_label=types, color=colors)
        plt.ylabel("Number of occurrences")
        plt.xlabel("Label")
        plt.show()
    
    def boxplot(self, col_true, col_false):
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        # Pandas dataframe
        data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

        # Plot the dataframe
        ax = data[['Reliable', 'Fake']].plot(kind='box', title='<NUM> count')

        # Display the plot
        plt.show()

def num_count():
    nc = count_number_occurences_per_label()
    pp.apply_pipeline("../datasets/tokenized-100k.csv", [
        (nc, None, "<num> count"),
        (pp.Binary_labels(), "type")
    ], new_file="../datasets/tokenized-100k-new.csv"
    )
    df = pp.apply_pipeline("../datasets/tokenized-100k-new.csv", [
    ],
    batch_size=100000,
    get_batch=True
    )
    nc.print_stats()
    #nc.plot_stats()
    #nc.plot_stats_binary()
    false = df[df["type"] == False]['<num> count']
    true = df[df["type"] == True]['<num> count']
    print(false)
    nc.boxplot(true, false)

#Clean data while keeping punctuation
patterns_with_punctuation = {
            #Urls
            re.compile(r'((https?:\/\/)?(?:www\.)?[a-zA-Z0-9-_\+=.:~@#%]+\.[a-zA-Z0-9()]{1,6}\b(?:[a-zA-Z0-9-_.:\\/@#$%&()=+~?]*))'): ' <URL> ',
            re.compile(r'(https?:\/\/)?w{0,3}\.?[a-z]+\.[a-z]\w*[\w\/-]*'): ' <URL> ',
            #Dates
            re.compile(r'(\d{1,2}([\:\-/\\]|(,\s)?)){2}\d{2,4}|\d{2,4}(([\:\-/\\]|(,\s)?)\d{1,2}){2}'): ' <DATE> ',
            re.compile(r'([Jj]an(uary)?|[Ff]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)([\:\-/\\]|(,\s)?)\d{1,2}([\:\-/\\]|(,\s)?)\d{1,4}'): ' <DATE> ',
            #Email
            re.compile(r'([\w.\-]+@(?:[\w-]+\.)+[\w-]{2,4})|@[\w\d]+'): ' <EMAIL> ',
            #Tabs and new lines
            re.compile(r'(\r\n|\n|\r)+'): ' ',
            re.compile(r'(\t+)'): ' ',
            #Punctuation
            re.compile(r'\!'): ' ! ',
            re.compile(r'\?'): ' ? ',
            #Other symbols
            re.compile(r'(\[|\])'): '',
            re.compile(r'[^A-Za-z0-9\s<>!?]'): '',
            #Numbers
            re.compile(r'(\d+)(th)?'): ' <NUM> ',
            #Spaces
            re.compile(r'( +)'): ' ',
        }
class Clean_data_punct(pp.FunctionApplier):
    def function_to_apply(self, cell):
        # Apply patterns using list comprehension
        cell = str(cell)
        cell = cell.lower()
        # Loop through each pattern and apply the pattern to each row and do replacement if needed
        for pattern, replacement in patterns_with_punctuation.items():
            cell = re.sub(pattern, replacement, cell)

        return cell

def ist_pipeline_punct():
    stopwords_lst = pp.stopwords.words('english')
    pp.apply_pipeline("../datasets/clean-100k.csv", [ 
        (Clean_data_punct(), "content"),
        (pp.Tokenizer(), "content"),
        (pp.Remove_stopwords(stopwords_lst), "content"),
        (pp.Stem(), "content"),
    ], new_file="../datasets/tokenized-100k.csv")

#Count punctuation occurences
class count_punctuation(pp.FunctionApplier):
    def __init__(self):
        keys = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 'satire', 'state', 
                'reliable', 'clickbait', 'political', 'rumor']
        self.typeDict = {k: 0 for k in keys}

        #[question mark count, exclamation mark count, combined]
        self.count_comb_dict = {k: [0, 0, 0] for k in keys}
        self.perTypeDict = {}
        self.binaryDict = {'Reliable': [0, 0, 0], 'Fake': [0, 0, 0]}

    def function_to_apply(self, row):
        try:
            excl_count = row['content'].count("!")
            q_count =  row['content'].count("?")
            counts = [q_count, excl_count, q_count + excl_count]
            self.count_comb_dict[row['type']] = [sum(x) for x in zip(counts, self.count_comb_dict[row['type']])]
            self.typeDict[row['type']] += 1
        except:
            pass

        for k, v in self.count_comb_dict.items():
            try: 
                self.perTypeDict[k] = [x / self.typeDict[k] for x in v]
            except: 
                self.perTypeDict[k] = [0, 0, 0]
        return q_count + excl_count

    def print_stats(self):
        print(self.perTypeDict)
        
    def plot_stats(self):
        types = list(self.perTypeDict.keys())
        counts = list(self.perTypeDict.values())
        x = np.arange(len(types))
        y1 = []
        y2 = []
        y3 = []
        for elm in counts:
            y1.append(elm[0])
            y2.append(elm[1])
            y3.append(elm[2])
        width = 0.2
        plt.xticks(x, types)

        plt.bar(x-0.2, y1, width, color='cyan')
        plt.bar(x, y2, width, color='orange')
        plt.bar(x+0.2, y3, width, color='green')
        plt.xlabel("Number of occurrence")
        plt.ylabel("Label")
        plt.legend(["Question marks", "Exclamation marks", "Combined"])
        plt.show()
    
    def plot_stats_binary(self):
        true_types = ['reliable', 'political', 'clickbait']
        for type in list(self.perTypeDict.keys()):
            if type in true_types:
                self.binaryDict['Reliable'] = [sum(x) for x in zip(self.perTypeDict[type], self.binaryDict['Reliable'])]
            else:
                self.binaryDict['Fake'] = [sum(x) for x in zip(self.perTypeDict[type], self.binaryDict['Fake'])]
        types = list(self.binaryDict.keys())
        counts = list(self.binaryDict.values())
        x = np.arange(len(types))
        y1 = []
        y2 = []
        y3 = []
        for elm in counts:
            y1.append(elm[0])
            y2.append(elm[1])
            y3.append(elm[2])
        width = 0.2
        plt.xticks(x, types)

        plt.bar(x-0.2, y1, width, color='lightsalmon')
        plt.bar(x, y2, width, color='darkcyan')
        plt.bar(x+0.2, y3, width, color='blue')
        plt.ylabel("Number of occurrences")
        plt.xlabel("Label")
        plt.legend(["Question marks", "Exclamation marks", "Combined"])
        plt.show()

    def boxplot(self, col_true, col_false):
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        # Pandas dataframe
        data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

        # Plot the dataframe
        ax = data[['Reliable', 'Fake']].plot(kind='box', title='\'?\' and \'!\' count')

        # Display the plot
        plt.show()

def punct_count():
    pc = count_punctuation()
    pp.apply_pipeline("../datasets/1mio-raw-cleaned-punct.csv", [
        (pc, None, "punct count"),
        (pp.Binary_labels(), "type")
    ], new_file="../datasets/punct.csv"
    )
    df = pp.apply_pipeline("../datasets/punct.csv", [
    ],
    batch_size=100000,
    get_batch=True
    )
    false = df[df["type"] == False]['punct count']
    true = df[df["type"] == True]['punct count']
    #pc.print_stats()
    #pc.plot_stats()
    #pc.plot_stats_binary()
    pc.boxplot(true, false)

#Religious content occurrence
class religious_content(pp.FunctionApplier):
    def __init__(self):
        #religious_words = ["chapel", "priest", "preacher", "nun", "saint", "st", 
        #                   "holy", "pray", "hymn", "soul"]
        religious_words = ["i"]
        ps = PorterStemmer()
        self.stem_religious_words = []
        for w in religious_words:
            self.stem_religious_words.append(ps.stem(w))

        keys = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 'satire', 
                'reliable', 'clickbait', 'political', 'rumor']
        self.count_dict = {k: 0 for k in keys}
        self.typeDict = {k: 0 for k in keys}
        self.perTypeDict = {}
        self.binaryDict = {'Reliable': 0, 'Fake': 0}

    def function_to_apply(self, row):
        try:
            sum = 0
            for word in row['content'].split(", "):
                if word[1:-1] in self.stem_religious_words:
                    sum += 1
                    self.count_dict[row['type']] += 1
            self.typeDict[row['type']] += 1
        except:
            pass

        for k, v in self.count_dict.items():
            try: 
                self.perTypeDict[k] = v / self.typeDict[k]
            except: 
                self.perTypeDict[k] = 0
        return sum

    def print_stats(self):
        print(self.perTypeDict)
        
    def plot_stats(self):
        types = list(self.perTypeDict.keys())
        counts = list(self.perTypeDict.values())
        colors = ['purple', 'purple','purple', 'purple', 'purple', 'purple', 'purple', 'green', 'green', 'green']
        plt.bar(range(len(self.perTypeDict)), counts, tick_label=types, color=colors)
        plt.xlabel("Number of occurrences")
        plt.ylabel("Label")
        plt.show()
    
    def plot_stats_binary(self):
        true_types = ['reliable', 'political', 'clickbait']
        for type in list(self.perTypeDict.keys()):
            if type in true_types:
                self.binaryDict['Reliable'] += self.perTypeDict[type]
            else:
                self.binaryDict['Fake'] += self.perTypeDict[type]
        types = list(self.binaryDict.keys())
        counts = list(self.binaryDict.values())
        colors = ['green', 'purple']
        plt.bar(range(len(self.binaryDict)), counts, tick_label=types, color=colors)
        plt.ylabel("Number of occurrences")
        plt.xlabel("Label")
        plt.show()

    def boxplot(self, col_true, col_false):
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        # Pandas dataframe
        data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

        # Plot the dataframe
        ax = data[['Reliable', 'Fake']].plot(kind='box', title='Religious word count')

        # Display the plot
        plt.show()
    
def religious_count():
    rc = religious_content()
    pp.apply_pipeline("../datasets/1mio-raw-cleaned-punct.csv", [
        (rc, None, "Occurences of \'i\'"),
        (pp.Binary_labels(), "type")
    ], new_file="../datasets/1k_i.csv"
    )
    df = pp.apply_pipeline("../datasets/religious1000.csv", [],
        batch_size=1078, 
        get_batch=True)
    rc.print_stats()
    rc.plot_stats()
    rc.plot_stats_binary()
    true = df[df["type"] == True]["Religious words count"]
    false = df[df["type"] == False]["Religious words count"]
    rc.boxplot(true, false)
    
class content_length(pp.FunctionApplier):
    def __init__(self):
        keys = ['fake', 'conspiracy', 'junksci', 'hate', 'unreliable', 'bias', 'satire', 
                'reliable', 'clickbait', 'political', 'rumor']
        self.len_dict = {k: 0 for k in keys}
        self.typeDict = {k: 0 for k in keys}
        self.perTypeDict = {}
        self.binaryDict = {'Reliable': 0, 'Fake': 0}
        
    
    def function_to_apply(self, row):
        try:
            #print(len(row['content']))
            self.len_dict[row['type']] += len(row['content'])
            self.typeDict[row['type']] += 1
        except:
            pass

        for k, v in self.len_dict.items():
            try: 
                self.perTypeDict[k] = v / self.typeDict[k]
            except: 
                self.perTypeDict[k] = 0
        return len(row['content'])

    def print_stats(self):
        print(self.perTypeDict)

    def plot_stats(self):
        types = list(self.perTypeDict.keys())
        counts = list(self.perTypeDict.values())
        colors = ['purple', 'purple','purple', 'purple', 'purple', 'purple', 'purple', 'green', 'green', 'green']
        plt.bar(range(len(self.perTypeDict)), counts, tick_label=types, color=colors)
        plt.xlabel("Length")
        plt.ylabel("Label")
        plt.show()
    
    def plot_stats_binary(self):
        true_types = ['reliable', 'political', 'clickbait']
        for type in list(self.perTypeDict.keys()):
            if type in true_types:
                self.binaryDict['Reliable'] += self.perTypeDict[type]
            else:
                self.binaryDict['Fake'] += self.perTypeDict[type]
        types = list(self.binaryDict.keys())
        counts = list(self.binaryDict.values())
        colors = ['green', 'purple']
        plt.bar(range(len(self.binaryDict)), counts, tick_label=types, color=colors)
        plt.ylabel("Length")
        plt.xlabel("Label")
        plt.show()
    
    def boxplot(self, col_true, col_false):
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        # Pandas dataframe
        data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

        # Plot the dataframe
        ax = data[['Reliable', 'Fake']].plot(kind='box', title='Content length')

        # Display the plot
        plt.show()

def content_len():
    cl = content_length()
    pp.apply_pipeline("../datasets/1mio-raw-cleaned-punct.csv", [
        (cl, None, "content length"),
        (pp.Binary_labels(), "type")
    ], new_file="../datasets/content-len-1000.csv"
    )
    df = pp.apply_pipeline("../datasets/content-len-1000.csv", [], 
        batch_size=1078,
        get_batch=True
    )
    #cl.print_stats()
    #cl.plot_stats()
    #cl.plot_stats_binary()
    true = df[df["type"] == True]["content length"]
    false = df[df["type"] == False]["content length"]
    
    cl.boxplot(true, false)

class sentiment(pp.FunctionApplier):
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.sum_pos = 0
        self.sum_neg = 0

    def function_to_apply(self, row):
        verdict = self.sentiment_pipeline(row[0:512])
        return verdict[0]['label']

    def pie_chart(self, col_true, col_false):
        true = len(col_true)
        true_pos = len(col_true[col_true["sentiment"] == 'POSITIVE'])/true
        true_neg = len(col_true[col_true["sentiment"] == 'NEGATIVE'])/true

        false = len(col_false)
        false_pos = len(col_false[col_false["sentiment"] == 'POSITIVE'])/false
        false_neg = len(col_false[col_false["sentiment"] == 'NEGATIVE'])/false
        print(col_true)

        print(true)
        print(false)
        

        print(col_true[col_true["sentiment"] == 'POSITIVE'])
        print(len(col_true[col_true["sentiment"] == 'NEGATIVE']))

        print(len(col_false[col_false["sentiment"] == 'POSITIVE']))
        print(len(col_false[col_false["sentiment"] == 'NEGATIVE']))

        fig = plt.figure(figsize=(4,3),dpi=144)
        ax1 = fig.add_subplot(121)
        ax1.pie([true_pos, true_neg], colors = ["green", "purple"])

        ax2 = fig.add_subplot(122)
        ax2.pie([false_pos, false_neg], colors = ["green", "purple"])

        #plt.ylabel("Frequency of positive/negative")
        #plt.xlabel("Label")
        plt.show()

    def boxplot(self, col_true, col_false):
        # Set the figure size
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        # Pandas dataframe
        data = pd.DataFrame({"Reliable": col_true, "Fake": col_false})

        # Plot the dataframe
        ax = data[['Reliable', 'Fake']].plot(kind='box', title='Content length')

        # Display the plot
        plt.show()

def sentiment_analysis():
    sa = sentiment()
    """pp.apply_pipeline("../datasets/1mio-raw-cleaned.csv", [
        (sa, "content", "sentiment"),
        (pp.Binary_labels(), "type")
    ], new_file="1k-sentiment.csv"
    )"""
    df = pp.apply_pipeline("1k-sentiment.csv", []
    , batch_size=1078, get_batch=True)
    true = df[df["type"] == True]
    false = df[df["type"] == False]
    sa.pie_chart(true, false)


if __name__ == "__main__":
    ist_pipeline_punct()


    #num_count()
    #punct_count()
    religious_count()
    #content_len()
    #sentiment_analysis()

    """file = "../datasets/clean-100k.csv"
    df = pd.read_csv(file)
    df['content length'] = content_length(df)
    true_types = ['reliable', 'political', 'clickbait']
    #binary()
    binary_file = "../datasets/clean-100k-binary.csv"
    df_binary = pd.read_csv(binary_file)
    df['binary label'] = df_binary['type']
    print(df['type'])
    print(df['binary label'])
    df_true = df.loc[str(df['binary label']) == 'True']
    df_false = df.loc[str(df['binary label']) == 'False']

    print(len(df_true))
    print(len(df_false))

    #print(df.dtypes)
    #df_true = df[df['type'] in true_types]
    #df_false = df[df['type'].any() not in true_types]"""
    
   


