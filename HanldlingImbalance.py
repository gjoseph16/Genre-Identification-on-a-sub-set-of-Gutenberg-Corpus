from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
from imblearn.over_sampling import SVMSMOTE
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler as SD




data=pd.read_csv (r'/Users/ramancheema/Desktop/Features25.csv')

oversample = SVMSMOTE(sampling_strategy='auto',k_neighbors=1)
X=data.iloc[:,2:27]
y=data.iloc[:,1]
X_res, y_res = oversample.fit_resample(X, y)  #this fit line is giving error

######################### till then is oversampling ##############################################

#ignore this below part , its for plotting purpose
##### plotting
le = preprocessing.LabelEncoder()
le.fit(y_res)
y=le.transform(y_res)

#get inverse
#list(le.inverse_transform(y))

#LDA reduction
lda = LDA(n_components=2) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_res, y))

#plotting stuff
plt.figure(figsize=(15,10))

plt.scatter(lda_transformed[y==0][0], lda_transformed[y==0][1], label='Allegories', c='red')
plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Christmas Stories', c='black')
plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Detective and Mystery', c='blue')
plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Ghost and Horror', c='lightgreen')
plt.scatter(lda_transformed[y==4][0], lda_transformed[y==4][1], label='Humorous and Wit and Satire', c='green')
plt.scatter(lda_transformed[y==5][0], lda_transformed[y==5][1], label= 'Literary', c='yellow')
plt.scatter(lda_transformed[y==6][0], lda_transformed[y==6][1], label= 'Love and Romance', c='cyan')
plt.scatter(lda_transformed[y==7][0], lda_transformed[y==7][1], label= 'Sea and Adventure', c='magenta')
plt.scatter(lda_transformed[y==8][0], lda_transformed[y==8][1], label= 'Western Stories', c='0.50')

# Display legend and show plot
plt.title('balanced Dataset- LDA redcued')
plt.legend(loc='upper right')
#plt.legend(loc=3)
plt.show()






