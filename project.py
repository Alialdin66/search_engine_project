
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from natsort import natsorted
import math
import pandas as pd
import numpy as np

stop_words = set(stopwords.words('english'))
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')
stemmer = PorterStemmer()
files_name = natsorted(os.listdir('DocumentCollection'))

document_of_terms = []
print('#------------------------------------------------read txt files------------------------------------------------#')

for document_name in files_name:
    with open(f'DocumentCollection/{document_name}', 'r') as f:
        document = f.read()
        print(document)#read txt files(1)
        
    #------------------------------------------stemming and tokenize-----------------------------------------#
    tokenized_documents = word_tokenize(document)

    terms = [stemmer.stem(word) for word in tokenized_documents if word not in stop_words]
    
    document_of_terms.append(terms)
print('#------------------------------------------------stemming and tokenize------------------------------------------------#')
print(document_of_terms)#read txt files after stemming and tokenize(1,2)

    #-----------------------------------------------Positional index-------------------------------------------------#
document_number=1
positional_index={}
for document in document_of_terms:

  for pos, term in enumerate(document):
        # print(pos, '-->' ,term)
        
        # If term already exists in the positional index dictionary.
        if term in positional_index:
                
            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1
            
            # Check if the term has existed in that DocID before.
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(pos)
                    
            else:
                positional_index[term][1][document_number] = [pos]

        # If term does not exist in the positional index dictionary
        else:
            
            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})     
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [pos]
 

    # Increment the file no. counter for document ID mapping             
 
  document_number += 1
print('#-----------------------------------------------Positional index-------------------------------------------------#')

print(positional_index)#Positional index(3)
 
    #-----------------------------------------------Pharseqeury----------------------------------------------#


def phrase_query(positional_index):
    query = input("Enter the query: ")
    final_list = [[] for _ in range(10)]

    for word in query.split():
     for key in positional_index[word][1].keys():
            if final_list[key - 1] != []:
                if final_list[key - 1][-1] == positional_index[word][1][key][0] - 1:
                    final_list[key - 1].append(positional_index[word][1][key][0])
            else:
                final_list[key - 1].append(positional_index[word][1][key][0])

    for position, positions_list in enumerate(final_list, start=1):
        if len(positions_list) == len(query.split()):
            print(f'doc:{position}')



# Assuming positional_index is available, you can call the function like this:
print('#-------------------------------------------Pharseqeury-----------------------------------------------------#')
#phrase_query(positional_index)#phrase_query(4)
# ---------------------------------------------tf--------------------------------------------
all_terms = []
for doc in document_of_terms:
    for term in doc:
        all_terms.append(term)


def get_tf(document):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document:
        wordDict[word] += 1
    return wordDict

tf = pd.DataFrame(get_tf(document_of_terms[0]).values(), index=get_tf(document_of_terms[0]).keys())
for i in range(1, len(document_of_terms)):
    tf[i] = get_tf(document_of_terms[i]).values()
tf.columns = ['doc'+str(i) for i in range(1, 11)]
print('------------------------------------------------------tf------------------------------------')
print(tf)# time frec(5)
# ----------------------------------------------------------wtf-----------------#
def weighted_tf(x):
    if x > 0:
        return math.log10(x) + 1
    return 0


for i in range(0, len(document_of_terms)):
    tf['doc'+str(i+1)] = tf['doc'+str(i+1)].apply(weighted_tf)

print('-------------------------------------------tfw------------------------------')
print(tf)
#------------idf----------------\
idf = pd.DataFrame(columns=['df', 'idf'])
for i in range(len(tf)):
    freq = tf.iloc[i].values.sum()

    idf.loc[i, 'df'] = int(freq)

    idf.loc[i, 'idf'] = math.log10(10 / (float(freq)))

idf.index=tf.index
print('-------------------------------------------idf------------------------------')

print(idf)

#------------idf*tf----------------\
tf_idf = tf.multiply(idf['idf'], axis=0)
print('-------------------------------------------idf*tf------------------------------')

print(tf_idf)

#-----------------doc-len-----------------#
def get_doc_len(col):
    return np.sqrt((tf_idf[col] ** 2).sum())

doc_len = pd.DataFrame(columns=['doc_len'])
for col in tf_idf.columns:
    doc_len.loc[col, 'doc_len'] = get_doc_len(col)

print('-------------------------------------------doc-len------------------------------')
print(doc_len)

#-----------------normalized-----------------#
norm_tf_idf = pd.DataFrame()
def get_norm_tf_idf(x, col):
    if doc_len.loc[col, 'doc_len'] != 0:
        return x / doc_len.loc[col, 'doc_len']
    else:
        return 0

for col in tf_idf.columns:
    norm_tf_idf[col] = tf_idf[col].apply(lambda x: get_norm_tf_idf(x, col))

print('-------------------------------------------doc-normalized------------------------------')
print(norm_tf_idf)


#------------------------------------insert qeury--------------------------#
print('-------------------------------------------insert_queryd------------------------------')
  
def get_w_tf(x):
    try:
        return math.log10(x)+1
    except:
        return 0
q='antoni brutu'
qeury=pd.DataFrame(index=norm_tf_idf.index)
qeury['tf']=[1 if x in q.split() else 0 for x in list(norm_tf_idf.index)]
qeury['w_tf']=qeury['tf'].apply(lambda x : get_w_tf(x))
product=norm_tf_idf.multiply(qeury['w_tf'],axis=0)

qeury['idf']=idf['idf'] *qeury['w_tf']
qeury['tf_idf']=qeury['w_tf']*qeury['idf']
qeury['norm']=0 
for i in range(len(qeury)):
    qeury['norm'].iloc[i]=float(qeury['idf'].iloc[i])/ math.sqrt(sum(qeury['idf'].values**2))#جزر جمع كل ال idf


print('-------------------------------------------info about:insert_query------------------------------')


print(qeury.loc[q.split()])

product2=product.multiply(qeury['norm'],axis=0)
scores={}
for col in product2.columns:
    if 0 in product2[col].loc[q.split()].values:
        pass
    else:
        scores[col]=product2[col].sum()


#-------------------------------------------------product(query*method docs)--------------------------#

print('-------------------------------------------product(query*method docs)------------------------------')
select_d1_d2=product2[list(scores.keys())].loc[(q.split())]
print(select_d1_d2)


print('----------------------------------------------------------------------------------')

doc_leng=math.sqrt(sum([x**2 for x in qeury['idf'].loc[q.split()]])) 
print('doc length:')
print(doc_leng)
print('----------------------------------------------------------------------------------')

column_sums=select_d1_d2.sum(axis=0)
print('column_sums:')

print(column_sums)
print('----------------------------------------------------------------------------------')

final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('returned_doc:')
for doc in final_score:
    returned_d = doc[0]
    print(f'{returned_d}', end=',')



