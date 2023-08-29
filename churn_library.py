# library doc string
"""
Supporting Functions for Predicting customer Churn notebook
Developed by: Shaik Sabiha
Version 1 Date: 21-08-2022
"""

# import libraries
import os
#import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.metrics import plot_confusion_matrix

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_clean_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    # read the file contents into the dataframe
    data_frame = pd.read_csv(pth)
    print("---- Data is imported ----\n")

    # Add the feature Churn
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    print("---- New Feature Churn is added ----\n")

    # drop irrelevant features
    data_frame.drop(['Unnamed: 0', 'Attrition_Flag', 'CLIENTNUM'], axis=1,
                    inplace=True)
    print("---- Irrelevant columns are dropped ----\n")

    # QC of the data
    folder_name = 'Data_QC'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # print(f"'{folder_name}' has been created.")
    else:
        # print(f"'{folder_name}' already exists.")
        pass
    print("---- Performing QC ----\n")

    # Check the nulls in the dataframe
    null_counts = data_frame.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    # Convert the columns_with_nulls Series to a DataFrame
    columns_with_nulls_data_frame = columns_with_nulls.reset_index()
    columns_with_nulls_data_frame.columns = ['Column Name', 'Null Count']
    print(f"1. {columns_with_nulls_data_frame.shape[0]} Columns have nulls")
    print("2. Details on Nulls is exported to the Data_QC/columns_with_nulls.csv")

    # Export to CSV
    columns_with_nulls_data_frame.to_csv('Data_QC/columns_with_nulls.csv',
                                         index=False)

    # export the descriptive statistics to a file
    data_frame.describe().to_csv('Data_QC/Data_stats.csv', index=False)
    print("3. Descriptive statistics is exported to the Data_QC/Data_stats.csv")

    # Convert the dtypes Series to a DataFrame
    dtypes_data_frame = data_frame.dtypes.reset_index()

    # Export to CSV with index name and column names
    dtypes_data_frame.to_csv('Data_QC/Data_dtypes.csv', index=False)
    dtypes_data_frame.columns = ['Column Name', 'Data Type']
    print("4. Column datatype info is exported to the Data_QC/Data_dtypes.csv\n")

    print("---- QC Done ----\n")


    # return the dataframe
    return data_frame


def plot_histogram(data_frame, column, folder_name):
    '''
    Plots a histogram for a given column of a DataFrame.

    input:
        data_frame: pandas DataFrame
        column: str, the name of the column to plot
        folder_name: str, the directory to save the plot

    output:
        JPEG image saved to folder
    '''
    # plot the histogram
    plt.figure(figsize=(20, 10))
    data_frame[column].hist(label=column)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{folder_name}/histogram_of_{column}.jpeg',
                format='jpeg', dpi=300)
    plt.show()
    plt.close()


def plot_barplot(data_frame, column, folder_name):
    '''
    Plots a bar plot for a given categorical column of a DataFrame.

    input:
        data_frame: pandas DataFrame
        column: str, the name of the column to plot
        folder_name: str, the directory to save the plot

    output:
        JPEG image saved to folder
    '''
    # plot the barplot of categorical column
    plt.figure(figsize=(20, 10))
    data_frame[column].value_counts('normalize').plot(kind='bar')
    plt.title(f'Barplot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'{folder_name}/histogram_of_{column}.jpeg',
                format='jpeg', dpi=300)
    plt.show()
    plt.close()


def plot_density(data_frame, column, folder_name):
    '''
    Plots a density plot for a given column of a DataFrame.

    input:
        data_frame: pandas DataFrame
        column: str, the name of the column to plot
        folder_name: str, the directory to save the plot

    output:
        JPEG image saved to folder
    '''
    # plot the density plot
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame[column], stat='density', kde=True)
    plt.title(f'Density plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'{folder_name}/Density plot of {column}.jpeg',
                format='jpeg', dpi=300)
    plt.show()
    plt.close()


def plot_correlation(data_frame, folder_name):
    '''
    Plots a correlation heatmap for numerical columns of a DataFrame.

    input:
        data_frame: pandas DataFrame
        folder_name: str, the directory to save the plot

    output:
        JPEG image saved to folder
    '''
    # plot the correlation heatmap
    numeric_data_frame = data_frame.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        numeric_data_frame.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.xlabel('Numeric Columns')
    plt.ylabel('Numeric Columns')
    plt.savefig(f'{folder_name}/Correlation Heatmap.jpeg',
                format='jpeg', dpi=300)
    plt.show()
    plt.close()


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            Jpeg images exported to the folder
    '''

    # Create eda folder
    folder_name = 'images/eda'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # print(f"'{folder_name}' has been created.")
    else:
        # print(f"'{folder_name}' already exists.")
        pass

    print("---- Performing EDA ----\n")

    # numerical columns list
    num_columns = data_frame.select_dtypes(exclude='object').columns.tolist()

    # categorical columns list
    cat_columns = data_frame.select_dtypes(include='object').columns.tolist()

    # Plotting numerical columns
    for col in num_columns:
        plot_histogram(data_frame, col, folder_name)

    # Plotting categorical columns
    for col in cat_columns:
        plot_barplot(data_frame, col, folder_name)

    # Density plot for 'Total_Trans_Ct'
    plot_density(data_frame, 'Total_Trans_Ct', folder_name)

    # Correlation plot
    plot_correlation(data_frame, folder_name)

    print("---- All images are exported to images/eda folder ----\n")


def encoder_helper(data_frame, category_lst, target_col):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: target column name

    output:
            data_frame: pandas dataframe with new encoded columns
    '''
    # make a copy of dataframe
    data_frame_copy = data_frame.copy()

    # encoding into a new columns
    for i in category_lst:
        lst = []
        groups = data_frame_copy.groupby(i).mean()[target_col]
        for val in data_frame_copy[i]:
            lst.append(groups.loc[val])
        tag = i + '_' + target_col
        data_frame_copy[tag] = lst

    # Drop the categorical features of the category_lst
    data_frame_copy.drop(category_lst, axis=1, inplace=True)

    return data_frame_copy


def perform_feature_engineering(data_frame, target_col):
    '''
    input:
              data_frame: pandas dataframe
              target: target column

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # List of categorical columns that need encoding
    category_lst = data_frame.select_dtypes(include='object').columns.tolist()

    # encode the categorical columns
    data_frame_encoded = encoder_helper(data_frame, category_lst, target_col)
    print("---- Categorical columns are Encoded ----\n")

    # Alternative approach
    # convert categorical features to dummy variable
    # data_frame_encoded = pd.get_dummies(data_frame, columns=category_lst,
    # drop_first=True, prefix=target_col)

    # Seggragate the independent and target variables
    y_df = data_frame_encoded[target_col]
    x_df = data_frame_encoded.drop(target_col, axis=1)

    # This cell may take up to 15-20 minutes to run
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3,
                                                        random_state=42)
    print("---- Dataset is split into train and test ----\n")
    return x_train, x_test, y_train, y_test

def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder

    input:
                    model_name: (str) name of the model, ie 'Random Forest'
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds: training predictions from model_name
                    y_test_preds: test predictions from model_name

    output:
                     None
    '''
    # Create eda folder
    folder_name = 'images/results'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # print(f"'{folder_name}' has been created.")
    else:
        # print(f"'{folder_name}' already exists.")
        pass

    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "images/results",
            fig_name),
        bbox_inches='tight',
        dpi=300)

    # Display figure
    plt.show()
    plt.close()


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder using plot_classification_report
    helper function

    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                     None
    '''
    # plot the classification report for LR
    plot_classification_report('Logistic Regression',
                               y_train,
                               y_test,
                               y_train_preds_lr,
                               y_test_preds_lr)

    plt.close()

    # plot the classification report for RF
    plot_classification_report('Random Forest',
                               y_train,
                               y_test,
                               y_train_preds_rf,
                               y_test_preds_rf)
    plt.close()


def feature_importance_plot(model, x_data, model_name, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                     None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name),
                bbox_inches='tight', dpi=300)

    # display feature importance figure
    plt.show()
    plt.close()


def confusion_matrix(model, model_name, x_test, y_test):
    '''
        Display confusion matrix of a model on test data

        input:
            model: trained model
            model_name: Name of the model
            X_test: X testing data
            y_test: y testing data
        output:
            None
        '''

    class_names = ['Not Churned', 'Churned']
    plt.figure(figsize=(15, 5))
    ax_plot = plt.gca()
    plot_confusion_matrix(model,
                          x_test,
                          y_test,
                          display_labels=class_names,
                          cmap=plt.cm.Blues,
                          xticks_rotation='horizontal',
                          colorbar=False,
                          ax=ax_plot)
    # Hide grid lines
    ax_plot.grid(False)
    plt.title(f'{model_name} Confusion Matrix on test data')
    plt.savefig(
        os.path.join(
            "images/results",
            f'{model_name}_Confusion_Matrix'),
        bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
                      X_train: X training data
                      X_test: X testing data
                      y_train: y training data
                      y_test: y testing data
    output:
                      None
    '''

    # Create eda folder
    folder_name = 'models'
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # print(f"'{folder_name}' has been created.")
    else:
        # print(f"'{folder_name}' already exists.")
        pass
    # Initialize Random Forest model
    rfc = RandomForestClassifier(random_state=42)

    # Initialize Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    # grid search for random forest parameters and instantiation
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Ramdom Forest using GridSearch
    cv_rfc.fit(x_train, y_train)
    print("---- Trained RandomForest Model ----\n")

    # Train Logistic Regression
    lrc.fit(x_train, y_train)
    print("---- Trained Logistic Regression Model ----\n")

    # get predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # calculate classification scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    print("----Classification reports exported to images/results folder----\n")

    # plot ROC-curves
    plt.figure(figsize=(15, 8))
    ax_plot = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=ax_plot,
        alpha=0.8
    )

    plot_roc_curve(lrc, x_test, y_test, ax=ax_plot, alpha=0.8)

    # save ROC-curves to images directory
    plt.savefig(
        os.path.join(
            "images/results",
            'ROC_curves.png'),
        bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    print("----ROC reports exported to images/results folder----\n")

    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')
    print("----Pickled best models saved to models folder----\n")

    for model, model_type in zip([cv_rfc.best_estimator_, lrc],
                                 ['Random_Forest', 'Logistic_Regression']
                                 ):
        # Display confusion matrix on test data
        confusion_matrix(model, model_type, x_test, y_test)

    print("----Confusion matrix exported to images/results folder----\n")

    # Display feature importance on train data
    feature_importance_plot(cv_rfc.best_estimator_,
                            x_train,
                            'Random_Forest',
                            "images/results")
    print("----Feature importance results exported to images/results folder----\n")


if __name__ == "__main__":
    dataset = import_clean_data("/data/bank_data.csv")
    print('Dataset successfully loaded')
    perform_eda(dataset)
    print('Finished EDA')
    train_X, test_X, train_y, test_y = perform_feature_engineering(
        dataset, target_col='Churn')
    print('Finished Feature Engg and Data split')
    train_models(train_X, test_X, train_y, test_y)
    print('Training completed. Best model weights + performance matrics saved')
