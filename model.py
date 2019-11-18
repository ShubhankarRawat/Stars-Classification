import pandas as pd

dataset = pd.read_csv('stars.csv')
df = dataset.copy()

df.info()
df.isnull().sum()

df.columns = ['temperature', 'luminosity', 'radius', 'am', 'type', 'color', 'class']

df.color.value_counts()

colors = {'Red': 'red', 'Blue': 'blue', 'Blue-white': 'blue white', 'Blue White': 'blue white', 
          'yellow-white': 'yellow white', 'White': 'white', 'Blue white': 'blue white', 'white': 'white',
          'Yellowish White': 'yellow white', 'yellowish': 'yellow', 'Orange': 'orange', 'Whitish': 'white',
          'Yellowish': 'yellow', 'Blue': 'blue', 'White-Yellow': 'yellow white', 'Blue white': 'blue white',
          'Blue-White': 'blue white', 'Orange-Red': 'orange', 'Pale yellow orange': 'orange'}

df.color = df.color.map(colors)
df.color.fillna('blue', inplace = True)
df.isnull().sum()

df['class'].value_counts()
df = pd.get_dummies(df, columns = ['class'], prefix = ['class'])
del df['class_G']

df['color'].value_counts()
df = pd.get_dummies(df, columns = ['color'], prefix = ['color'])
del df['color_yellow']

a = df['type']
del df['type']
df['type'] = a

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


import lightgbm as lgb

d_train = lgb.Dataset(X_train, label = y_train)
params = {}

clf = lgb.train(params, d_train, 100)
#Prediction
y_pred = clf.predict(X_test)
#convert into binary values
for i in range(0, len(y_pred)):
    if y_pred[i] <= 0.5:       # setting threshold to .5
       y_pred[i] = 0
    elif y_pred[i] <= 1.5 and y_pred[i] > 0.5:  
       y_pred[i] = 1
    if y_pred[i] <= 2.5 and y_pred[i] > 1.5:       # setting threshold to .5
       y_pred[i] = 2
    elif y_pred[i] <= 3.5 and y_pred[i] > 2.5:  
       y_pred[i] = 3
    elif y_pred[i] <= 4.5 and y_pred[i] > 3.5:  
       y_pred[i] = 4
    elif y_pred[i] <= 5.5 and y_pred[i] > 4.5:
        y_pred[i] = 5
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i] <= 0.5:       # setting threshold to .5
       y_pred_train[i] = 0
    elif y_pred_train[i] <= 1.5 and y_pred_train[i] > 0.5:  
       y_pred_train[i] = 1
    if y_pred_train[i] <= 2.5 and y_pred_train[i] > 1.5:       # setting threshold to .5
       y_pred_train[i] = 2
    elif y_pred_train[i] <= 3.5 and y_pred_train[i] > 2.5:  
       y_pred_train[i] = 3
    elif y_pred_train[i] <= 4.5 and y_pred_train[i] > 3.5:  
       y_pred_train[i] = 4
    elif y_pred_train[i] <= 5.5 and y_pred_train[i] > 4.5:
        y_pred_train[i] = 5
        
cm_train = confusion_matrix(y_pred_train, y_train)


s_train = 0
s_test = 0
for i in range(0, len(cm_test)):
    for j in range(0, len(cm_test)):
        if i == j:
            s_train = s_train + cm_train[i][j]
            s_test = s_test + cm_test[i][j]
            
print('Accuracy for train set for LightGBM = {}'.format(s_train/len(y_train)))
print('Accuracy for test set for LightGBM = {}'.format(s_test/len(y_test)))
