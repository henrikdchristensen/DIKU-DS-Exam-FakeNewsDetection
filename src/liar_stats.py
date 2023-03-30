#Sender
class subject(pp.FunctionApplier):
    def __init__(self):
        self.dict = {}
        self.sort_by_value_dict = {}

    def function_to_apply(self, cell):
        subject_list = str(cell).split(",")
        for subject in subject_list:
            if subject in self.dict:
                self.dict[subject] += 1
            else:
                self.dict[subject] = 1

    def sort_dict(self):
        self.sort_by_value_dict = dict(sorted(self.dict.items(), key=lambda x:x[1], reverse = True))

    def print_top_subjects(self, n):
        self.sort_dict()
        d = dict(list(self.sort_by_value_dict.items())[0: n])
        for key in list(d.keys()):
            print(key)

su = subject()
pp.apply_pipeline("../datasets/liar_dataset/combined_cleaned.csv", [
    (su, 'subjects')
], progress_bar=True
)
su.print_top_subjects(10)



def word_count(file, new_file):
    wc = s.word_count()
    pp.apply_pipeline(file, [
        (wc, None, "word count")
        #(pp.Binary_labels(), "type")
    ], new_file=new_file
    )
    df = pp.apply_pipeline(new_file, [], 
        batch_size=10239,
        get_batch=True
    )
    #print(df.dtypes)
    print(df["word count"].describe())

word_count("../datasets/liar_dataset/train.csv", "../datasets/liar_dataset/train_word_count.csv")



class party_affiliation(pp.FunctionApplier):
    def __init__(self):
        self.dict = {}
        self.sort_by_value_dict = {}

    def function_to_apply(self, cell):
        affiliation = str(cell).lower()
        if affiliation in self.dict:
            self.dict[affiliation] += 1
        else:
            self.dict[affiliation] = 1
    
    def sort_dict(self):
        self.sort_by_value_dict = dict(sorted(self.dict.items(), key=lambda x:x[1], reverse = True))

    def print_dict(self):
        self.sort_dict()
        print(self.sort_by_value_dict)

pa = party_affiliation()
pp.apply_pipeline("../datasets/liar_dataset/combined_cleaned.csv", [
    (pa, "party")
], progress_bar=True
)
pa.print_dict()