# library doc string
"""
Testing and logging Codes for Predicting customer Churn notebook
Developed by: Shaik Sabiha
Version 1 Date: 12-09-2023
"""

import os
import logging
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import
    '''
    try:
        data_frame = cls.import_clean_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    dataset = cls.import_clean_data("data/bank_data.csv")
    try:
        cls.perform_eda(dataset)
        logging.info("Testing perform_eda: SUCCESS")
    except AttributeError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err
    except SyntaxError as err:
        logging.error("Testing perform_eda: Input should be a dataframe")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    dataset = cls.import_clean_data("data/bank_data.csv")
    try:

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        cls.encoder_helper(dataset, cat_columns, 'Churn')

        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing encoder_helper: There are column names that doesn't exist in your dataframe")

    try:
        assert isinstance(cat_columns, list)
        assert len(cat_columns) > 0

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: category_lst argument should be a list with length > 0")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    dataset = cls.import_clean_data("data/bank_data.csv")
    try:

        target = 'Churn'

        cls.perform_feature_engineering(dataset, target)

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except KeyError as err:
        logging.error(
            "Testing perform_feature_engineering: Target column names doesn't exist in dataframe")

    try:
        assert isinstance(target, str)

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: response argument should str")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    dataset = cls.import_clean_data("data/bank_data.csv")
    train_X, test_X, train_y, test_y = cls.perform_feature_engineering(
        dataset, target_col='Churn')

    try:
        cls.train_models(train_X, test_X, train_y, test_y)
        logging.info("Testing train_models: SUCCESS")
    except MemoryError as err:
        logging.error(
            "Testing train_models: Out of memory while train the models")
        raise err


if __name__ == "__main__":

    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
