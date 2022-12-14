# Importing necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting
from sklearn.utils.random import sample_without_replacement
from pandas.api.types import is_numeric_dtype

# For undersampling and oversampling data
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, RandomUnderSampler

# For selecting best parameters for the model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# For Data Model Development
from sklearn import linear_model
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# For Neural Network Model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Kaggle dataset link
# https://www.kaggle.com/competitions/its-a-fraud/data

# Reading test and train csv files through pandas Dataframe
df = pd.read_csv("/content/drive/MyDrive/dataset/DataSet/train.csv")
test = pd.read_csv("/content/drive/MyDrive/dataset/DataSet/test.csv")

# Separating Y data from train dataset and combining the test and train datasets for pre-processing
Y_train = df.pop('isFraud')
dataframes_to_be_combined = [df,test]
df = pd.concat(dataframes_to_be_combined)

# Checking feature DeviceInfo
df['DeviceInfo'].describe()
df['DeviceInfo'].isna().sum()
df.drop("DeviceInfo", axis=1, inplace=True)

# Dropping one row from any two rows which are correlated greater than 95% 
cor = df.corr().abs()
upper = cor.where(np.triu(np.ones(cor.shape),k=1).astype(bool))
drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df.drop(df[drop], axis=1, inplace=True)
drop

# Removing all features where null values are greater than 50%
null = df.isnull().sum() / len(df)
missing_features = null[null > 0.50].index
df.drop(missing_features, axis=1, inplace=True)
missing_features

# Filling all columns with 25%ile value(median) where 25%ile and 75%ile values are same
for column in df.columns[1:]:
    if(df[column].dtype.kind in 'biufc' and df[column].quantile(0.75) == df[column].quantile(0.25)):
        df[column].fillna(df[column].quantile(0.25), inplace = True)

# Filling null values of few columns which are unique with the mean or median by checking their bar graphs and box plots
df["card2"].fillna(df["card2"].mean(), inplace = True)
df["card5"].fillna(df["card5"].median(), inplace = True)
df['addr1'].fillna(df['addr1'].median(), inplace=True)
df['P_emaildomain'].fillna('gmail.com', inplace=True)

# Filling all categorical data with their mode
for column in df.columns[1:]:
    if(df[column].isna().sum() > 0):
        if(df[column].dtype == 'object'):
            df[column].fillna(df[column].mode(dropna =True)[0], inplace = True)

# Filling null values of all the columns which have data that contains one value with a very high probability
# Checked these columns by plotting histograms and observing mean, median values and data segregation
df['D4'].fillna(df['D4'].median(), inplace=True)
df['D10'].fillna(df['D10'].median(), inplace=True)
df['D15'].fillna(df['D15'].median(), inplace=True)
df['V12'].fillna(df['V12'].mean(), inplace=True)
df['V13'].fillna(df['V13'].mean(), inplace=True)
df['card5'].fillna(df['card5'].median(), inplace=True)
df['D1'].fillna(df['D1'].median(), inplace=True)
df['D11'].fillna(df['D11'].median(), inplace=True)
df['V310'].fillna(df['V310'].median(), inplace=True)
df['V130'].fillna(df['V130'].median(), inplace=True)
df['V96'].fillna(df['V96'].median(), inplace=True)
df['V285'].fillna(df['V285'].median(), inplace=True)
df['V99'].fillna(df['V99'].median(), inplace=True)
df['V282'].fillna(df['V282'].median(), inplace=True)
df['V283'].fillna(df['V283'].median(), inplace=True)

# Filling all the null values of remaining columns with their mean values
for column in df.columns[1:]:
    if(df[column].isna().sum() > 0):
       df[column].fillna(df[column].mean(), inplace=True)

# Removing all the outliers to avoid data misleadings
class OutlierRemoval: 
    def __init__(self, lower_quartile, upper_quartile):
        self.lower = lower_quartile - 1.5*(upper_quartile - lower_quartile)
        self.upper = upper_quartile + 1.5*(upper_quartile - lower_quartile)
    def removeOutlier(self, x):
        return (x if x <= self.upper and x >= self.lower else (self.lower if x < self.lower else (self.upper)))
        
for column in df.columns[1:]:
    if(df[column].dtype.kind in 'biufc' and column != 'isFraud'):
      outlier_remover = OutlierRemoval(df[column].quantile(0.25), df[column].quantile(0.75))
      df[column] = df[column].apply(outlier_remover.removeOutlier)

# One hot encoding all the catgorical data
columns_e = []
for column in df.columns[1:]:
    if(df[column].dtype.kind not in 'biufc'):
        columns_e.append(column)
df = pd.get_dummies(df, columns=columns_e)
n = len(df)

# Separating test and train data after the completion of pre-processing
X_train = df.iloc[:(df.shape[0]-test.shape[0]),:]
X_test = df.iloc[(df.shape[0]-test.shape[0]):,:]

# Random oversampler and undersampler for increasing and decreasing samples
ros = RandomOverSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)
pipeline = Pipeline(steps=[('o', ros), ('u', rus)])

# Using only under sampler as it is giving better results through some observation
X_res, y_res = rus.fit_resample(X_train, Y_train)

## Hyperparameter optimization using RandomizedSearchCV for various models

''''
Logistic Regression Model: GridSearchCV

    C = np.logspace(0, 4, num=10)
    penalty = ['l1', 'l2']
    solver = ['liblinear', 'saga']
    hyperparameters = dict(C=C, penalty=penalty, solver=solver)
    logistic = linear_model.LogisticRegression()
    gridsearch = GridSearchCV(logistic, hyperparameters)
    best_model = gridsearch.fit(X_res, y_res)
'''

''''
Logistic Regression Model: RandomizedSearchCV

    C = np.logspace(0, 4, num=10)
    penalty = ['l1', 'l2']
    solver = ['liblinear', 'saga']
    hyperparameters = dict(C=C, penalty=penalty, solver=solver)
    logistic = linear_model.LogisticRegression()
    randomizedsearch = RandomizedSearchCV(logistic, hyperparameters)
    best_model_random = randomizedsearch.fit(X_res, y_res)
    logistic = linear_model.LogisticRegression(C=3593.813663804626, penalty='l1', solver='liblinear')
    logistic.fit(X_res, y_res)
    Y_pred = logistic.predict(X_test)
'''

'''
Naive Bayes Model: RandomizedSearchCV

    param_distributions_nb = {
        'priors': [None, [0.1,]*len(2),],
        'var_smoothing': np.logspace(0,-9, num=100)
    }
    nbModel = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=param_distributions_nb, verbose=1, cv=10, n_jobs=-1)
    best_model_random = nbModel.fit(X_res, y_res)
    guassian = GaussianNB(var_smoothing=3.5111917342151277e-07, priors=None)
    guassian.fit(X_res, y_res)
    Y_pred = guassian.predict(X_test)
'''

'''
Perceptron Model: RandomizedSearchCV

    mlp = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }

    clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    best_model_random = clf.fit(X_res, y_res)
    mlp = MLPClassifier(max_iter=100, alpha=0.05, learning_rate='adaptive', hidden_layer_sizes = (100,), activation = 'relu', solver = 'adam')
    mlp.fit(X_res, y_res)
    Y_pred = mlp.predict(X_test)
'''

'''
Decision Tree Model: RandomizedSearchCV

    params = {
        'max_depth': [55,58,60,63,65,68,70,72],
        'min_samples_split': [1,2,3,4],
        'min_samples_leaf': [13,15,17]
    }
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=57, min_samples_split=3, min_samples_leaf=15)
    clf.fit(X_res, y_res)
'''

'''
Keras Sequential Neural Network Model: Manual Hyperparameter Tuning

    # Parameters used for the Neural Network model
    lr = 0.5
    hidden_layer_act = 'relu'
    output_layer_act = 'sigmoid'
    no_epochs = 100

    # Scaling down the features for better perfomance of the model
    sc = StandardScaler()
    X_res_sc = sc.fit_transform(X_res)
    X_test_sc = sc.transform(X_test)

    # Sequential neural network model
    model = Sequential()

    # Adding multiple layers to the Sequential model
    model.add(Dense(152, input_dim=228, activation=hidden_layer_act))
    model.add(Dense(64, activation=hidden_layer_act))
    model.add(Dense(32, activation=hidden_layer_act))
    model.add(Dense(1, activation=output_layer_act))
    print(model.summary())

    # Training the model with train dataset
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X_res_sc, Y_res, epochs=no_epochs, batch_size = 2048,  verbose = 2)
    predictions = model.predict(X_res_sc)
    rounded = [int(round(x[0])) for x in predictions]
    predictions.flatten()

    # Calculating train accuracy
    my_accuracy = accuracy_score(Y_res, predictions.round())
    print(my_accuracy)

    # Predicting Y for test dataset and storing it in a csv file
    predictions = model.predict(X_test_sc)
    predictions.flatten()
    predictions = np.round(predictions)
    Y_test = pd.DataFrame(predictions, columns = ['isFraud'])
    Y_test.index.name = "Id"
    Y_test.to_csv("/content/NN_fraud.csv")
'''

# Hyper Parameter Optimization for XGBoost Classifier
params = {
 "learning_rate"    : [0.20] ,
 "max_depth"        : [12, 14, 16, 18],
 "min_child_weight" : [1, 2, 3, 4],
 "colsample_bytree" : [ 0.8],
 "gamma" : [0.2]
}

classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_res,y_res)

# XGBoost classifier with best parameters
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.2, learning_rate=0.02,
       max_delta_step=0, max_depth=12, min_child_weight=2, missing=-1,
       n_estimators=2000, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1, eval_metric='auc')

# Fitting the model with the undersampled data
classifier.fit(X_res, y_res)

# Predicting the Y for test data and storing it in csv file for submission
Y_pred = classifier.predict(X_test)
Y_test = pd.DataFrame(Y_pred)
Y_test.columns = ['isFraud']
Y_test.index.name = "Id"
Y_test.to_csv("/content/fraud.csv") # saving the csv file