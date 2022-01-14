import textstat
import pandas as pd
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
from collections import Counter
import spacy
from textstat.textstat import textstatistics, legacy_round


merged_corpus= pd.read_csv (r'/Users/ramancheema/Desktop/dataset.csv')


#appending features in new dataframe
features=[]
for i in range(0, 994):
        text = merged_corpus.iloc[i, 4]
        book_id = merged_corpus.iloc[i, 1]

        #F1
        # number of words
        word_count = textstat.lexicon_count(text, removepunct=True)

        #F2
        # punc count ratio
        punc_count = (textstat.lexicon_count(text, removepunct=False))
        punc_count = (punc_count - word_count) / word_count

        # number of sentences
        doc = nlp(text)
        sent = doc.sents
        sentences = list(doc.sents)
        num_sent = len(sentences)

        #F4
        #avg syllable
        syllable = textstat.syllable_count(text)
        avg_syllables_per_word = float(syllable) / float(word_count)

        # poly syll count
        count = 0
        words = []
        for sentence in sent:
            words += [token for token in sentence]

        for word in words:
            syllable_count = textstat.syllable_count(str(word))
            if syllable_count >= 3:
                count += 1

############################ READING EASE #####################################

        #F5
        # flesch reading ease
        FRE = 206.835 - float(1.015 * average_sentence_length) - float(84.6 * avg_syllables_per_word)

        #F6
        #smog reading ease
        if num_sent >= 3:
            SMOG = (1.043 * (30 * (count / num_sent)) ** 0.5) 
                   + 3.1291
        else:
            SMOG = 0

############################ SENTIMENT##########################################

        # feature sentiment analysis
        s = TextBlob(text)
        senti = s.sentiment
        # F7
        polarity = senti[0]
        # F8
        subjectivity = senti[1]

############################ SENTENCE LEVEL COMPLEXITY ############################
        # sentence level Complexity- avg sentence and sentence complexity

        #F3
        # avg sentence length
        average_sentence_length = float(word_count / num_sent)

        # using POS tagging to get all other features

        tokens = word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        counts = Counter(tag for word, tag in tags)

         #no.of tokens
        tok=len(tokens)
        if(tok==0):
           tok=1

        # F9
        # sentence complexity
        clause = counts['CC']
        num_sent = sentence_count(text)
        sen_complex = clause / num_sent

############################ LEXIACL RICHNESS/COMPLEXITY ############################
        #  F10 lexical diversity
        # no. of unique words/length of text
        lex_rich = len(set(tokens)) / tok

    #  F11 lexical density (feature 11,12,13,14)
        nouns=counts['NN']+counts['NNS']+counts['NNP']+counts['NNPS']
        noun_ratio=(nouns)/tok

    #F12
        verbs=counts['VB']+counts['VBD']+counts['VBG']+counts['VBZ']+counts['VBP']
        verb_ratio=(verbs)/tok

    #F13
        adverbs=counts['RB'] + counts['RBR'] + counts['RBS'] + counts['WBR'] + counts['RBS']
        adverb_ratio=(adverbs)/tok

    #F14
        adjcs=counts['JJ']+counts['JJR']+counts['JJS']
        adjective_ratio=(adjcs)/tok

############################ WRITING STYLE #######################################

    #F15 - personal pronoun
        peronal_pron = counts['PRP']/tok

    #F16   feature writing style- possesive pronoun
        possesive_pron = counts['PRP$']/tok

    #F17    feature writing style - proposition
        proposition = counts['IN']/tok

    # F18   feature  writing style - colon , hypen and semicolon
    # covers ; - :
        colo_hyph= counts[':']/tok   #counts[',']

    #F19  writing style - Interjection
        inter=counts['UH']/tok

    #F20 writing style - punctuation subordination conjunction
        p_sub_conj=counts['CC']/tok

############################ COUNTRY-SIDE/URBAN SETTING ######################################

        #number of characters and qoutes

       # F21 double qoutes
        quote = counts['``'] / tok

        #F22 number of characters
        unique_tokens = set(tokens)
        new_tags = nltk.pos_tag(unique_tokens) #counting each character only one
        unique_counts = Counter(tag for word, tag in new_tags)
        proper_noun=unique_counts['NNP']

############################ HIGH-LEVEL RATIO ######################################


    #F23
        adv_adjc=adverb_ratio/adjective_ratio

    #F24
        adjc_noun=adjective_ratio/noun_ratio

    #F25
        pron = counts['PRP'] + counts['WP'] + counts['PRP$']
        pron_ratio = (pron) / tok
        adjc_pron=adjective_ratio/pron_ratio




    #append GENRE too
        genre=merged_corpus.iloc[i,2]
        feat = [book_id,genre,word_count,punc_count,average_sentence_length ,avg_syllables_per_word ,FRE,SMOG,polarity,
            subjectivity,sen_complex,lex_rich,noun_ratio,verb_ratio,adverb_ratio,adjective_ratio,
            peronal_pron,possesive_pron,proposition,colo_hyph,inter,p_sub_conj,quote,proper_noun,
            adv_adjc,adjc_noun,adjc_pron]
        features.append(feat)

features=pd.DataFrame(features, columns= ['book_id','genre','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15',
                                                                    'F16','F17','F18','F19','F20','F21','F22','F23','F24','F25'])

features.to_csv(r'/Users/ramancheema/Desktop/Features25.csv', index=False)





#while combining files
#three= pd.read_csv (r'/Users/ramancheema/Desktop/features_pycharm2.csv')
#seven=pd.read_csv (r'/Users/ramancheema/Desktop/features_PART1.csv')
#sixteen=pd.read_csv (r'/Users/ramancheema/Desktop/features_pycharm3.csv')
#m=pd.merge(left=seven, right=three, left_on='book_id', right_on='book_id')
#del three['genre']
#del sixteen['genre']
#merge37=pd.merge(left=seven, right=three, left_on='book_id', right_on='book_id')
#merge37['f3_y']=merge37['f3_y']/merge37['f3_x']
#del merge37['f3_x']
