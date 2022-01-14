from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

data= pd.read_csv (r'/Users/ramancheema/Desktop/Features25.csv')
train,test= train_test_split(data,test_size=.20,shuffle=True,stratify=data['genre'])
X=train.iloc[:,2:27]
y=train.iloc[:,1]
scaler = MinMaxScaler()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)


parameter_candidates = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, cv=10, scoring='f1_micro', n_jobs=-1)

# Train the classifier
clf.fit(X, y)

print('Best score for data:', clf.best_score_)
# View the best parameters for the model found using grid search
print('Best C:',clf.best_estimator_.C)
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)

#Best score for data: 0.20787981956888557
#Best C: 100
#Best Kernel: linear
#Best Gamma: scale

x_test = scaler.transform(test.iloc[:,2:27])
y_test=test.iloc[:,1]
y_test=le.transform(y_test)
plot_confusion_matrix(clf,x_test, y_test)  # doctest: +SKIP
plt.show()

clf.score(testm, y)  #0.1869687181230874




