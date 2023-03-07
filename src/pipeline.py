import h5py
import filehandling as fh
import preprocessing


class FunctionApplier:
    def function_to_apply(self, row):
        pass


class Count_rows(FunctionApplier):

    def __init__(self):
        self.rows = 0

    def function_to_apply(self, row):
        self.rows += 1
        return row

class Clean_data(FunctionApplier):

    def __init__(self):
        pass

    def function_to_apply(self, row):
        clean
        return row

def apply_pipeline(old_file, functions, new_file=""):
    with h5py.File(old_file, 'r') as f:
        data_set = f['data']
        if new_file != "":

            save_to = h5py.File(new_file, 'w')

            arr = fh.create_empty_string_array(data_set.shape[1])
            save_to = save_to.create_dataset('data', data=arr, maxshape=(
                None, data_set.shape[1]), dtype=h5py.string_dtype(encoding='utf-8'))
            save_to[0] = data_set[0, ]

        for i in range(data_set.shape[0]):
            output = data_set[i, ]
            for func in functions:
                output = func.function_to_apply(output)

            if new_file != "":
                save_to.resize((i+2, data_set.shape[1]))
                data_set[-len(output):] = output

        try:
            save_to.close()
        except:
            pass


func = Count_rows()
apply_pipeline("../datasets/sample/data.h5", [func])
print(func.rows)
