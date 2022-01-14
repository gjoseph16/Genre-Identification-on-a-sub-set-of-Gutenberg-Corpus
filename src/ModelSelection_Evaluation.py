from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import  svm
import matplotlib.pyplot as plt


#use stratification coz we want contribution of each class in test and validation data
#also keep 20% data aside for test but use stratify=data['genre'] , this we get way some from  all  categories
#use k fold validation , it will automatically take care of least populated class in y

data=pd.read_csv (r'/Users/ramancheema/Desktop/Features25.csv')

X=data.iloc[:,2:27]
y=data.iloc[:,1]
#le = preprocessing.LabelEncoder()
#le.fit(y_res)
#y=le.transform(y_res)

scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X_res)

X, y = shuffle(X, y, random_state=42)
train,test= train_test_split(X,y,test_size=.20,shuffle=True,stratify=y['genre'])

################## Model Selection code #######################
models = [
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  f_score = cross_val_score(model, X, y, scoring='f1_micro', cv=CV)
  for fold_idx, f in enumerate(f_score):
    entries.append((model_name, fold_idx, f))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1_micro'])
import seaborn as sns
sns.boxplot(x='model_name', y='f1_micro', data=cv_df)
sns.stripplot(x='model_name', y='f1_micro', data=cv_df,
              size=8, jitter=True, edgecolor="magenta", linewidth=2)
plt.title('Imbalance Dataset ')
plt.show()


##############################Balanced Dataset on SVM #####################################
oversample = SMOTE(sampling_strategy='auto',k_neighbors=1)
X=data.iloc[:,2:27]
y=data.iloc[:,1]
X_res, y_res = oversample.fit_resample(X, y)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_res)
le = preprocessing.LabelEncoder()
le.fit(y_res)
y=le.transform(y_res)

x,y,m,n= train_test_split(X, y,test_size=.20,shuffle=True,stratify=y)

parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates,cv=5,scoring='f1_micro', n_jobs=-1)

# Train the classifier
clf.fit(x, m)

y = scaler.transform(y)
n=le.transform(n)
clf.score(y, n)

plot_confusion_matrix(clf,y, n)
plt.title('Confusion Matrix- Balanced Dataset')
plt.show()

