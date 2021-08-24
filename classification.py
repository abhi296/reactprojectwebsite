import pandas as pd
import numpy as np
import ast
from configparser import ConfigParser
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

config = ConfigParser()
config.read(['configuration.ini','classification.ini'])
df = pd.read_csv(config.get('client_code_data', 'data_filename'))
df = df.set_index('id')


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

threshold = config.getfloat('threshold_numeric_categorical_division', 'threshold')

numeric = []
categorical = []

threshold = threshold
for i in X.columns:
    actual = X[i].nunique()
    if X[i].dtype == object or actual <= threshold:
        categorical.append(i)
    elif X[i].dtype == np.float64 or X[i].dtype == np.int64:
        numeric.append(i)
    else:
        continue

#Train-Test-val Split

df1 = df[numeric]
df2 = df[categorical]

test_per = int(config.get('test_percentage', 'test_per'))
split = test_per/100

val_per = int(config.get('val_percentage', 'val_per'))
split1 = val_per/100
X_1, X_val, y_1, y_val = train_test_split(X, y, test_size=split1, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=split, random_state=42)

X_train_copy = X_train.copy()
X_val_copy = X_val.copy()



# Missing Value Treatment

#print(X_train.columns[X_train.isnull().any()])  # Getting names of columns having null values
#print(X_train.isnull().sum())  # Getting the count of missing values in each column

for i in X_train.columns:
    X_train[i] = X_train[i].fillna(X_train[i].median())


for i in X_test.columns:
    X_test[i] = X_test[i].fillna(X_test[i].median())


for i in X_val.columns:
    X_val[i] = X_val[i].fillna(X_val[i].median())

#IQR(outlier treatment)

X_copy1 = X_train
#print(X.shape)
Q1 = X_train[df1.columns].quantile(0.25)
Q3 = X_train[df1.columns].quantile(0.75)
IQR = Q3 - Q1

if [X_train < (Q1 - 1.5 * IQR)]:
    s = Q1 - 1.5 * IQR
    for index, value in s.items():
        X_copy1[index] = X_copy1[index].mask(X_copy1[index] < value, value)
if [X_train > (Q3 + 1.5 * IQR)]:
    s = Q3 + 1.5 * IQR
    for index, value in s.items():
        X_copy1[index] = X_copy1[index].mask(X_copy1[index] > value, value)
X_train = X_copy1

X_copy2 = X_test

Q1 = X_test[df1.columns].quantile(0.25)
Q3 = X_test[df1.columns].quantile(0.75)
IQR = Q3 - Q1

if [X_test < (Q1 - 1.5 * IQR)]:
    s = Q1 - 1.5 * IQR
    for index, value in s.items():
        X_copy2[index] = X_copy2[index].mask(X_copy2[index] < value, value)
if [X_test > (Q3 + 1.5 * IQR)]:
    s = Q3 + 1.5 * IQR
    for index, value in s.items():
        X_copy2[index] = X_copy2[index].mask(X_copy2[index] > value, value)
X_test = X_copy2

X_copy1 = X_val

Q1 = X_val[df1.columns].quantile(0.25)
Q3 = X_val[df1.columns].quantile(0.75)
IQR = Q3 - Q1

if [X_val < (Q1 - 1.5 * IQR)]:
    s = Q1 - 1.5 * IQR
    for index, value in s.items():
        X_copy1[index] = X_copy1[index].mask(X_copy1[index] < value, value)
if [X_val > (Q3 + 1.5 * IQR)]:
    s = Q3 + 1.5 * IQR
    for index, value in s.items():
        X_copy1[index] = X_copy1[index].mask(X_copy1[index] > value, value)
X_val = X_copy1

#Label Encoding

categorical_feature_mask = X.dtypes == object
categorical_cols = X.columns[categorical_feature_mask].tolist()
le = LabelEncoder()
for col in categorical_cols:
    n = X_train[col]
    X_train[col] = le.fit_transform(n)
    X_test[col] = le.transform(n)
    X_val[col] = le.transform(n)

#MinMax Scaling

mm = MinMaxScaler()
X_train[df1.columns] = mm.fit_transform(X_train[df1.columns])
X_test[df1.columns] = mm.transform(X_test[df1.columns])
X_val[df1.columns] = mm.transform(X_val[df1.columns])

#SelectKbest

new_features = []
def my_score(X_train, y_train):
    return mutual_info_classif(X_train, y_train, random_state=42)
selk = SelectKBest(score_func=my_score, k=10)

column_names = list(X_train.columns)
X_train = selk.fit_transform(X_train, y_train)
# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(selk.inverse_transform(X_train))
selected_columns = list(selected_features.columns[selected_features.var()!=0])

for i in range(len(selected_columns)):
    selected_columns[i]=column_names[selected_columns[i]]     
X_test = selk.transform(X_test)
X_val = selk.transform(X_val)
X_train = pd.DataFrame(X_train, columns=selected_columns)
X_test = pd.DataFrame(X_test,columns=selected_columns)
X_val = pd.DataFrame(X_val, columns=selected_columns)

X_train1 = X_train.copy()
y_train1 = y_train.copy()

#Logistic Regression Classifier

print("Logistic Regression\n")

logreg = LogisticRegression(multi_class='multinomial', random_state=1)

search_grid = ast.literal_eval(config.get('Grid_Search_Parameters', 'lr_param_grid'))
lr_grid = GridSearchCV(estimator=logreg, param_grid=search_grid,
                       scoring='accuracy',
                       verbose=0, cv=5,
                       return_train_score=True)

lr_grid.fit(X_train, y_train)

y_pred1 = lr_grid.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred1)*100
f1_score_lr = f1_score(y_test, y_pred1, average='weighted')*100
params_lr = lr_grid.best_params_
print("Accuracy_score : ", accuracy_lr)
print("F1_score : ", f1_score_lr)
print("Grid_params : ", params_lr)
'''
import numpy as np
from skmultiflow.drift_detection import DDM

ddm = DDM()

#data_stream = np.random.randint(2, size=2000)
#data_stream
X_test = X_test.to_numpy()


# Changing the data concept from index 999 to 1500, simulating an 
# increase in error rate
for i in range(50,150):
    #X_test[i] = 0
    X_test[i] = y_pred1[i]
# Adding stream elements to DDM and verifying if drift occurred
for i in range(180):
    #ddm.add_element(y_test == y_pred1)
    ddm.add_element(X_test[i])
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(X_test[i]) + ' - of index: ' + str(i))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(X_test[i]) + ' - of index: ' + str(i))'''

#Random Forest Classifier

print("\n\nRandom Forest Classifier \n")
rfc = RandomForestClassifier(random_state=42)

search_grid = ast.literal_eval(config.get('Grid_Search_Parameters', 'rf_param_grid'))

rf_grid = GridSearchCV(estimator=rfc, param_grid=search_grid,
                       cv=5, n_jobs=-1, return_train_score=True)

rf_grid.fit(X_train1, y_train1)

y_pred2 = rf_grid.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred2)*100
f1_score_rf = f1_score(y_test, y_pred2, average='weighted')*100
params_rf = rf_grid.best_params_

print("Accuracy_score : ", accuracy_rf)
print("F1_score : ", f1_score_rf)
print("Grid_params : ", params_rf)
