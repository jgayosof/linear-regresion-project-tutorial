# imports:
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# import dataset & save:
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
df_raw.to_csv('data/raw/medical_insurance_cost.csv')

# create df_interim:
df_interim = df_raw.copy()


# convert categoricals to numericals:

# sex:
df_interim['sex'] = df_interim['sex'].apply(lambda x: 1 if x == 'male' else 0)
df_interim['sex'].value_counts()

# region:
def conv_region(region) :
    if region == 'southwest' :
        return 1
    elif region == 'southeast' :
        return 2
    elif region == 'northwest' :
        return 3
    elif region == 'northeast' :
        return 4
    else :
        return 'BadRegion'
df_interim['region'] = df_interim.apply(lambda x: conv_region(x['region']), axis=1)
df_interim['region'].value_counts()

# smoker
df_interim['smoker'] = df_interim['smoker'].apply(lambda x: 1 if x == 'yes' else 0)


# create df_processed and save:
df_processed = df_interim.copy()
df_interim.to_csv('data/interim/medical_insurance_cost_interim.csv')


# apply model (linear regression):
X = df_processed.drop(['charges'], axis=1)
y = df_processed['charges']

# train-test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# model:
LR = linear_model.LinearRegression()
LR.fit(X_train, y_train)

# run predictions:
y_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)
print(f'coefficient of determination: {LR.score(X_test,y_test)}')

# run an example:
edad = 33
sex = 1
bm = 22
children = 0
smoker = 1
region = 3
testing_y = np.array([edad, sex, bm, children, smoker, region])
#testing_y = testing_y.reshape(-1, 1)

print(f'Example: {testing_y}')
print('Predicted charges for example:', LR.predict([[edad, sex, bm, children, smoker, region]]))

#save model:
joblib.dump(LR, 'models/lr_ensurance.pkl')

