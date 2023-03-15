import numpy as np
import src.filehandling as fh

FILENAME = "../datasets/sample/news_sample.csv"
ROWS = 250
ROWS_PR_ITERATION = 200000


def test_read_rows():
    rows = fh.read_rows(filename=FILENAME, idx=2, num=2)
    assert rows.iloc[0, 3] == 'unreliable'
    assert rows.iloc[1, 2] == 'awm.com'


def test_create_randomly_split_array():
    new_size = 100
    arr = fh.create_randomly_split_array(old_size=ROWS, new_size=new_size, split=(0.8, 0.1, 0.1))
    assert len(arr) == ROWS
    assert np.sum(arr == fh.Set.TRAIN) == new_size*0.8
    assert np.sum(arr == fh.Set.VALI) == new_size*0.1
    assert np.sum(arr == fh.Set.TEST) == new_size*0.1