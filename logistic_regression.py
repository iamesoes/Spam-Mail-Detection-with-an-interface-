import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import chardet
#Data collection and Preproccessing
#loading data from csv file to a pandas dataframe
file ='spam.csv'

with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

df= pd.read_csv(file,encoding='Windows-1252')

#print(df)

#df.sample(5)


df.shape


##Steps
#1. Cleaning the data
#2. Exploratory data analysis
#3. Text preproccessing
#4. Model building
#5. Evaluation

df.info()

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

df.sample(5)

df.rename(columns={'v1':'Category','v2':'Text'},inplace=True)
df.sample(5)


encoder=LabelEncoder()

df.loc[df['Category']=='spam','Category',]=0
df.loc[df['Category']=='ham','Category',]=1

df.sample(5)


#seperating the data as texts and label

X = df['Text']
Y = df['Category']

#print(X)

#print(Y)


#Splitting the data into training data & test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)


#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)

#we will transform the text data to feature vectors that can be used as input to the Logistic Regression Model
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
#convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# print(X_train_features)

# # Training the model

# # Logistic Regression

model = LogisticRegression()

#training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# # Evaluating the trained model


# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)



print('Accuracy on training data : ',accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


print('Accuracy on test data : ',accuracy_on_test_data)


# # Building a Predictive System 


# input_mail=["Free entry in 2 a wkly comp to win."]
# #converting text to feature vectors
# input_data_features = feature_extraction.transform(input_mail)
# #making prediction
# prediction = model.predict(input_data_features)
# print(prediction)

def predict_spam(mail):
    input_data_features = feature_extraction.transform(mail)
    prediction = model.predict(input_data_features)

    if prediction == 1:
        return "HAM",accuracy_on_test_data, accuracy_on_training_data
    return "SPAM",accuracy_on_test_data, accuracy_on_training_data    
    


