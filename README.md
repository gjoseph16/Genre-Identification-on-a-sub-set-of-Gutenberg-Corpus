# Genre-Identification-on-a-sub-set-of-Gutenberg-Corpus
Gutenberg Corpus is a set of books belonging to the 19th Century English Fiction. This dataset is created from Project Gutenberg. 
It consists of about 1000 books and roughly 10 genres. We observed that each data point in this classification is a fiction book with a label (genre). 
So the primary question would be trying to detect (i.e. classify) the genre of a book. 
The Gutenberg Corpus consists of a set of books belonging to the 19th Century English Fiction. It  is created from Project Gutenberg consisting of about 
1000 books and roughly 10 genres. The English version of the dataset contains some of the HTML documents. Supervised Genre identification is a problem 
and our goal is to extract the features which are relevant to fiction books and investigate which supervised machine learning methods are best suited 
to solve it. We observed that this is a multi-class text classification problem, given a new book we must assign it to one of the 9 genres. The classifier 
would make the assumption that it is assigned to one and only one genre.

PROBLEM DESCRIPTION AND ASSUMPTIONS:
Gutenberg Corpus is a set of books belonging to the 19th Century English Fiction. This dataset is created
from Project Gutenberg. It consists of about 1000 books and roughly 10 genres. We observed that each
data point in this classification is a fiction book with a label (genre). So the primary question would be
trying to detect (i.e. classify) the genre of a book.

TEXT PRE-PROCESSING:
The Text processing pipeline begins by removing punctuations,
numbers, stop words and whitespace which are not
included in the analysis of the dataset. Next, all the
words are converted to lower-case only. Stemming
to be performed to syntactically match 2 words by
reducing to the base word. A technique would be
proposed to represent frequency term but sparsity
across the entire corpus.

FEATURE REDUCTION:
 Feature Selection:
As a feature selection method, i proposed to try out Filter methods such as Chi-Squared, Mutual
Information filter or maybe check which one performs better. But without actually
experimenting it with the dataset it is hard to tell which would give us the best result.
 Feature Extraction:
Made an effort to implement the following to extract the features but could
change in future according to the results.
- TF-IDF
- Word embedding
- Word2vec
- LDA (Linear Discriminant Analysis)

MODEL TRAINING:
Implemented the following classifiers to train a final model and may try other models too.
- Naïve Bayes classifier
- Linear classifier
- Support vector machine
- Deep Neural network

EVALUATION:
Considered the following to evaluate classification results:
- Accuracy
- F-Score
- Confusion Matrices

References
WORSHAM, J. M. (2014). TOWARDS LITERARY GENRE IDENTIFICATION: APPLIED NEURAL. Colorado
