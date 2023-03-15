import numpy as np
import src.filehandling as fh

DATA_FILENAME = "files/news_sample.csv"
TRAIN_FILENAME = "files/train.csv"
VALI_FILENAME = "files/vali.csv"
TEST_FILENAME = "files/test.csv"
OLD_SIZE = 250
NEW_SIZE = 200
ROWS_PR_ITERATION = 200000
SPLIT = (0.8, 0.1, 0.1)


def test_read_rows():
    rows = fh.read_rows(filename=DATA_FILENAME, idx=2, num=2)
    assert rows.iloc[0, 3] == 'unreliable'
    assert rows.iloc[1, 2] == 'awm.com'


def test_create_randomly_split_array():
    arr = fh.create_randomly_split_array(old_size=OLD_SIZE, new_size=NEW_SIZE, split=SPLIT)
    assert len(arr) == OLD_SIZE
    assert np.sum(arr != 0) == NEW_SIZE
    assert np.sum(arr == fh.Set.TRAIN) == NEW_SIZE*0.8
    assert np.sum(arr == fh.Set.VALI) == NEW_SIZE*0.1
    assert np.sum(arr == fh.Set.TEST) == NEW_SIZE*0.1
    
    
def test_number_of_rows():
    assert fh.number_of_rows(filename=DATA_FILENAME, rows_pr_iteration=ROWS_PR_ITERATION) == OLD_SIZE
    
def test_create_train_vali_and_test_sets():
    fh.create_train_vali_and_test_sets(old_size=OLD_SIZE, new_size=NEW_SIZE, split=SPLIT, data_filename=DATA_FILENAME, train_filename=TRAIN_FILENAME, vali_filename=VALI_FILENAME, test_filename=TEST_FILENAME, rows_pr_iteration=ROWS_PR_ITERATION)
    assert fh.number_of_rows(filename=TRAIN_FILENAME, rows_pr_iteration=ROWS_PR_ITERATION) == NEW_SIZE*0.8
    assert fh.number_of_rows(filename=VALI_FILENAME, rows_pr_iteration=ROWS_PR_ITERATION) == NEW_SIZE*0.1
    assert fh.number_of_rows(filename=TEST_FILENAME, rows_pr_iteration=ROWS_PR_ITERATION) == NEW_SIZE*0.1