import h5py


class FunctionApplier:
    def printname(self):
        print(self.firstname, self.lastname)


class Student(Person):
    def __init__(self, fname, lname):
        Person.__init__(self, fname, lname)


def apply_pipeline(old_file, functions, new_file=""):
    pass


def write_hdf_cols(filename: str, idx: int = 0, num: int = 1) -> np.ndarray:
    with h5py.File(filename, 'r') as f:
        return f['data'][:, idx:idx+num]
