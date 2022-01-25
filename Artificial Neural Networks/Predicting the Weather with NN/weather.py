import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from numpy.random import seed
import datetime
import pdb

def print_time():
    print("%s" % datetime.datetime.now())
    


# =========================================================

# Rain header
# Date,Location,MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustDir,WindGustSpeed,
# WindDir9am,WindDir3pm,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,
# Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday,RISK_MM,RainTomorrow

#exclude = ['Location', 'Date', 'RISK_MM']
exclude = ['Date', 'RISK_MM']
bools = ['RainToday', 'RainTomorrow']
categorical = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

df = pd.read_csv('weatherAUS.csv')

# Reduce size of dataset
df = df.loc[df['Location']=='Perth',:]  # Pick just one location, otherwise wind direction will be meaningless

# Remove unwanted variables
for var in exclude:
    del df[var]

# Boolean variables to {0,1}
for var in bools:
    df[var] = df[var].astype('category')
    df[var] = df[var].cat.codes      # convert to category codes

# Categorical variables to indicator variables, one new column per value
for var in categorical:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
        del df[var]

cols = list(df.columns.values)
cols.remove('RainTomorrow')
cols.append('RainTomorrow') # Move to the end
df = df[ cols ]


# =========================================================

seed(0)
print_time()


X = df.iloc[:,0:-1]
X = X.dropna(axis=1, how='all') # Drop totally empty columns here explicitly, to capture column names (impossible if Imputer drops cols)
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Dealing with missing data
col_names = list(X_train)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)


print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)

# =========================================================


# Ref: https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
#
# Rescale data for ANN
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Use the same scaler as on training set

classifier = MLPClassifier(hidden_layer_sizes=(50,),max_iter=50000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print_time()

