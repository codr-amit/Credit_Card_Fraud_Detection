import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the credit_card_data to the pandas dataframe
credit_card_data = pd.read_csv('creditcard.csv')

# first 5 rows of the dataset
print(credit_card_data.head())

# dataset informations
print(credit_card_data.info())

# number of rows and columns in the dataset
print(credit_card_data.shape)

# statistical information about the dataset
print(credit_card_data.describe())

# checking the missing values in each columns
print(credit_card_data.isnull().sum())

# distribution of legit and fraudulent transactions
print(credit_card_data['Class'].value_counts())                                 # this line tells us that the dataset is highly unbalanced because it have very less datapoints of fraudulent transactions 

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measures of these data
print(legit.Amount.describe())
print(fraud.Amount.describe())

#compare the transaction for both transactions
print(credit_card_data.groupby('Class').mean())

# Under-Sampling : Build a sample dataset containing similar distribution of normal transactions and fraudulent transactions
legit_sample = legit.sample(n=492)

# Concatenating two DataFrames
new_dataset = pd.concat([legit_sample,fraud],axis=0)
print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())


# Splitting the data into Features and Targets
X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']

# Split the data into Training data and Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


###########  Model Training  ##############

model = LogisticRegression()

# training the model with Training data
model.fit(X_train, Y_train)


##############  Model Evaluation  ########

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)

# accuracy score on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(testing_data_accuracy)



