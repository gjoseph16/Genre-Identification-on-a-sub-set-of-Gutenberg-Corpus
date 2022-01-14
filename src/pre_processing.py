import pandas as pd
import glob
import os

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
#importing csv file into dataframe separated by columns

#if you dont use latin-1 encoding ; it cant read coz of special characters and all other stuff
#if you dont put delimiter=; then all data comes under one column rather thatn 4 column
master996 = pd.read_csv (r'/Users/ramancheema/Documents/ notes/AML/Project/Gutenberg_English_Fiction_1k/master996.csv',
                         encoding='latin-1',delimiter=';')

#keeping book id value same in both .html and master996 same, for inner join further , dropping extra stuff
for i in range(0,master996.shape[0]):
    id,split,extra = (master996.iloc[i, 1]).partition('.') #bookid is 1st column, name is like 'pg10067.epub', we dont want text after '.'
    master996.iloc[i, 1]=id  # updated master996

file_list = glob.glob(os.path.join(os.getcwd(),
                "/Users/ramancheema/Documents/ notes/AML/Project/"
                "Gutenberg_English_Fiction_1k/Gutenberg_19th_century_English_Fiction",
                 "*.html"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        f = os.path.basename(f_input.name) #f='pg13985-content.html'
        n, s, e = f.partition('-')
        #getting read html tags
        soup = BeautifulSoup(f_input.read(), "html.parser")
        f=soup.get_text()
        c=[n,f]
        corpus.append(c)
corpus=pd.DataFrame(corpus, columns= ['book_id','text'])
merged_corpus= pd.merge(left=master996, right=corpus, left_on='book_id', right_on='book_id')
# step 1 still we got \n in text
merged_corpus['text'] = merged_corpus['text'].str.replace('[\n]', ' ')
merged_corpus['text'].head()

# step 1  Remove blank rows if any.
merged_corpus['text'].dropna(inplace=True)
merged_corpus['text'].head()

# STEP 2 Removing Punctuation
merged_corpus['text'] = merged_corpus['text'].str.replace('[^\w\s]', '')
merged_corpus['text'].head()

# step 3 Change all the text to lower case, so that CAR is same as car
merged_corpus['text'] = [entry.lower() for entry in merged_corpus['text']]
merged_corpus['text'].head()

# Step 4 Tokenization : each word is token now
merged_corpus['text'] = [word_tokenize(entry) for entry in merged_corpus['text']]
merged_corpus['text'].head()

# do tokinsation first, coz it make string to list, we can make comparison in stop word removal
# step 5 stopword removal
# make list for comaprison
sw = list(stopwords.words('english'))
ps = PorterStemmer()
for i in range(0, 996):
    s_words = merged_corpus.iloc[i, 4]
    stem_words = [ps.stem(w) for w in s_words]
    merged_corpus.iloc[i, 4] = stem_words

# step 6 stemming
ps = PorterStemmer()
for i in range(0, 996):
    s_words = merged_corpus.iloc[i, 4]
    stem_words = [ps.stem(w) for w in s_words]
    merged_corpus.iloc[i, 4] = stem_words
# ta da da

# saving dataframe to csv file
merged_corpus.to_csv(r'/Users/ramancheema/Desktop/Name.csv', index=False)

#step 6 stemming
#ps = PorterStemmer()
#for entry in merged_corpus['text']
#    for word in entry:
 #       ps.stem(word)

#merged_corpus['text']= [ps.stem(word) for word in entry for entry in merged_corpus['text']]
#merged_corpus['text'].head()





#from pandas import ExcelWriter

#writer = ExcelWriter('/Users/ramancheema/Desktop/t.xlsx')
#merged_corpus.to_excel(writer)
#writer.save()

#for entry in merged_corpus['text']
   #make a copy of the word_list
 # for word in entry: # iterate over word_list
 #   if word in stopwords.words('english'):
      # entry.remove(word)
