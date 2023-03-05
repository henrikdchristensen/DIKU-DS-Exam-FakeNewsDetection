import numpy as np
import src.filehandling as fh

csv_filename = "datasets/sample/news_sample.csv"
hdf_filename = "tests/files/data.h5"

COLS = 16
ROWS = 251
ROWS_PR_ITERATION = 200000


def test_num_of_cols_csv():
    assert fh.num_of_cols_csv(filename=csv_filename) == COLS


def test_csv_to_hdf():
    fh.csv_to_hdf(csv_filename=csv_filename, hdf_filename=hdf_filename,
                  cols=COLS, rows_pr_iteration=ROWS_PR_ITERATION)
    assert fh.num_of_rows_and_cols_hdf(
        filename=hdf_filename) == (ROWS, COLS)


def test_read_hdf_rows():
    rows = fh.read_hdf_rows(filename=hdf_filename,
                            idx=2, num=2)
    assert rows[0, 3] == bytes('fake', 'utf-8')
    assert rows[1, 2] == bytes('cnnnext.com', 'utf-8')


def test_create_randomly_split_array():
    arr = fh.create_randomly_split_array(size=ROWS-1, split=(0.8, 0.1, 0.1))
    assert len(arr) == ROWS-1
    assert np.sum(arr == fh.Set.TRAIN) == (ROWS-1)*0.8
    assert np.sum(arr == fh.Set.VALI) == (ROWS-1)*0.1
    assert np.sum(arr == fh.Set.TEST) == (ROWS-1)*0.1


def test_create_train_vali_and_test_sets():
    split = fh.create_randomly_split_array(size=ROWS-1, split=(0.8, 0.1, 0.1))
    data_filename = "tests/files/data.h5"
    fh.csv_to_hdf(csv_filename=csv_filename, hdf_filename=data_filename,
                  cols=COLS, rows_pr_iteration=ROWS_PR_ITERATION)
    train_filename = "tests/files/train.h5"
    vali_filename = "tests/files/vali.h5"
    test_filename = "tests/files/test.h5"
    fh.create_train_vali_and_test_sets(split=split, cols=COLS, data_filename=data_filename, train_filename=train_filename, vali_filename=vali_filename, test_filename=test_filename,
                                       rows_pr_iteration=ROWS_PR_ITERATION)
    assert fh.num_of_rows_and_cols_hdf(
        filename=train_filename) == (int((ROWS-1)*0.8+1), COLS)
    assert fh.num_of_rows_and_cols_hdf(
        filename=vali_filename) == (int((ROWS-1)*0.1+1), COLS)
    assert fh.num_of_rows_and_cols_hdf(
        filename=test_filename) == (int((ROWS-1)*0.1+1), COLS)
