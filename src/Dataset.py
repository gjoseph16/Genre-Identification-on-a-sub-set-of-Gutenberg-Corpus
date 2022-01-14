import pandas as pd
import numpy as np
import glob
import os
pre_data = pd.read_csv (r'/Users/ramancheema/Documents/ notes/AML/Project/Gutenberg_English_Fiction_1k/Pre_Processed.csv',
                         encoding='latin-1')
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
        f = os.path.basename(f_input.name)
        n, s, e = f.partition('-')
        #getting read html tags
        soup = BeautifulSoup(f_input.read(), "html.parser")
        f=soup.get_text()
        c=[n,f]
        corpus.append(c)
corpus=pd.DataFrame(corpus, columns= ['book_id','text'])

merged_corpus= pd.merge(left=master996, right=corpus, left_on='book_id', right_on='book_id')
# saving dataframe to csv file
merged_corpus.to_csv(r'/Users/ramancheema/Desktop/dataset.csv', index=False)
