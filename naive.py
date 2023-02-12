import pandas as pd
#Data collection and Preproccessing
#loading data from csv file to a pandas dataframe
file ='spam.csv'
import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

df= pd.read_csv(file,encoding='Windows-1252')
df.head()


# # Data Cleaning

df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.sample(5)

df.rename(columns={'v1':'Category','v2':'Text'},inplace=True)
df.sample(5)


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


#labeling spam mail as 0; ham mail as 1

df.loc[df['Category']=='spam','Category',]=0
df.loc[df['Category']=='ham','Category',]=1

#seperating the data as texts and label

X = df['Text']
Y = df['Category']


#print(X)

#print(Y)

#Splitting the data into training data & test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

#print(X.shape)
#print(X_train.shape)
#print(X_test.shape)


# # Feature Extraction

from sklearn.feature_extraction.text import TfidfVectorizer
#we will transform the text data to feature vectors that can be used as input to the Naive Bayes Model
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
#convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# # Training the model

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model = MultinomialNB()
#training the Naive Bayes model with the training data
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


# input_mail=["Hey!What about come to me tonight? We can watch movie together."]
# #converting text to feature vectors
# input_data_features = feature_extraction.transform(input_mail)
# #making prediction
# prediction = model.predict(input_data_features)
# print(prediction)


# input_mail=["OK, I'm waiting you at the Central Park"]
# #converting text to feature vectors
# input_data_features = feature_extraction.transform(input_mail)
# #making prediction
# prediction = model.predict(input_data_features)
# print(prediction)

# input_mail=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18"]
# #converting text to feature vectors
# input_data_features = feature_extraction.transform(input_mail)
# #making prediction
# prediction = model.predict(input_data_features)
# print(prediction)

def predict_spam_2(input_mail):
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if prediction == 1:
        return "HAM",accuracy_on_test_data, accuracy_on_training_data
    return "SPAM",accuracy_on_test_data, accuracy_on_training_data 
