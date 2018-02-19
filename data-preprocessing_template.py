import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset

dataset= pd.read_csv('Data.csv')
x= dataset.iloc[:, 0:3].values
dx= pd.DataFrame(x)
y= dataset.iloc[:, 3].values
dy= pd.DataFrame(y)

# Filling Up the missing values
from sklearn.preprocessing import Imputer
imputer_fill= Imputer(missing_values= 'NaN',strategy= "mean", axis= 0)
imputer_fill= imputer_fill.fit(x[:, 1:3])
x[:,1:3]= imputer_fill.transform(x[:,1:3])
dx= pd.DataFrame(x)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x= LabelEncoder()
x[:, 0]= labelencoder_x.fit_transform(x[:,0])
onehot= OneHotEncoder(categorical_features= [0])
x= onehot.fit_transform(x).toarray()
dx= pd.DataFrame(x)

labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)
dy= pd.DataFrame(y)

# Splitting the dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 0)

# Scaling the values
from sklearn.preprocessing import StandardScaler
st_scale= StandardScaler()
x_train= st_scale.fit_transform(x_train)
x_test= st_scale.transform(x_test)
