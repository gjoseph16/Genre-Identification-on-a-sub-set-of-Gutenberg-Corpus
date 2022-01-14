from sklearn import preprocessing
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

data=pd.read_csv (r'/Users/ramancheema/Desktop/Features25.csv')
X=data.iloc[:,2:27]
y=data.iloc[:,1]

le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)

#scaling of X (0,1)

scaler = MinMaxScaler()
scaler.fit(X)
S=scaler.transform(X)    #chi works for non negative and also if i dont scale down ,
                          # it picks the features with highest range like word count so normalize every one of it

############### UNIVARIATE SELECTION- SelectKBest Chi square#############################
bestfeatures = SelectKBest(score_func=chi2, k=25)
fit = bestfeatures.fit(S,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(25,'Score'))  #print 10 best features

######################## Feature Importance##########################
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.title('Feature Importance-Tree Based Classifiers (with feature scaling)')
plt.show()

######################## Correlation Matrix with Heatmap ##########################
#was not very helpful
#data = pd.read_csv(r'/Users/ramancheema/Desktop/Features25.csv')
#X = data.iloc[:,2:27]  #independent columns
#y = data.iloc[:,1]    #target column i.e price range
# #get correlations of each features in dataset
#corrmat = data.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()
