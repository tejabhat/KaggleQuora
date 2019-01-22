# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:54:20 2019

@author: tejbhat

This class does following - 
    1) normalize the lines:
        a)POS tag the words using nltk.
        
        Replace the person names with "PERSON_NAME" and org names with "ORG_NAME".
        We do not want to process the person/org names, as our model shouldn't bias based on person/org names.
        
        Also, if it is a tree of continuous words, join those with "_". for ex. TBD
        
        b) if a work contains negations, like "shouldn't create" , convert that into "not_create"
        
        c)Lemmatize using nltk, and create a new line with this output.
    
    2) remove non-alphabetic chars
    
    3) tokenize based on space char only
    
    
"""

#SOS_token="SOS"#start of sentence token
#EOS_token="EOS"#end of sentence token

"""
TBD
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
          'stanford-ner.jar')
text="where is india"
tagging = st.tag(text.split()) 
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')

import re

class SentencePreProcessor():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Improving the stop words list
        self.stop_words = stopwords.words('english')
        #uncheck_words = ["don","won","doesn","couldn","isn","wasn","wouldn","can","ain","shouldn","not"]
        self.negation_words = ["don't", "won't","doesn't","couldn't","isn't","wasn't","wouldn't","can't","ain't","shouldn't","not"]
        #does becomes doe if we lemmatize
        self.dont_lemmat=["does"]
        #g_org_names=["ebay","amazon"]
        
    def tokenize_line(self,line):
        arr=[]
        #sent=self.normalizeString(sent)            
        #arr.append(SOS_token)
        for word in line.split(' '):
            arr.append(word)
        #arr.append(EOS_token)
        return arr
        
    def tokenize_lines(self,sentences):
        ret=[]
        
        for sent in sentences:
            arr=self.tokenize_line(sent)
            ret.append(arr)
        return ret
    
    
    def continuous_chunks(self,line):
        chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(line)))
        continuous_chunk = []
        person_names=[]
        org_names=[]
        for i in chunked:
            ttype=i.__class__.__name__
            if ttype == "Tree":
                if i.label()== "PERSON":
                    person_names.append(" ".join([token for token, pos in i.leaves()]))
                elif i.label()== "ORGANIZATION":
                    org_names.append(" ".join([token for token, pos in i.leaves()]))
                else:
                    if len(i.leaves()) > 1:
                        continuous_chunk.append(" ".join([token for token, pos in i.leaves()]))
            
            elif ttype == "Tuple":
                if i[1] == "PERSON":
                    person_names.append(i[0])
                elif i[1] == "ORGANIZATION":                    
                    org_names.append(i[0])
            
                 #elif current_chunk:
                 #        named_entity = " ".join(current_chunk)
                 #        if named_entity not in continuous_chunk:
                 #                continuous_chunk.append(named_entity)
                 #                current_chunk = []
                 #else:
                 #        continue
        return person_names, org_names, continuous_chunk
    
        
        """chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(line)))
        prev = None
        continuous_chunk = []
        current_chunk = []
        for i in chunked:
            #print(type(i))
            if type(i) == nltk.Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
                else:
                    continue
        return continuous_chunk 
        """
    def normalize_lines(self,lines):
        new=[]
        for line in lines:
            line=self.normalize_line(line)
            new.append(line)
        return new
        
    def normalize_line(self,line):
        p_names,org_names,chunks=self.continuous_chunks(line)
        for chunk in chunks:
            tmp=re.sub(" ","_",chunk)
            try:
                line=re.sub(chunk, tmp, line)
            except Exception as e:
                print("chunk=",chunk,str(e) )
        for chunk in p_names:
            try:
                line=re.sub(chunk, "PERSON_NAME", line)
            except Exception as e:
                print("chunk=",chunk,str(e) )
            
        for chunk in org_names:
            try:
                line=re.sub(chunk, "ORG_NAME", line)
            except Exception as e:
                print("chunk=",chunk,str(e) )                
            
            
        line = line.lower()
        word_list = line.split(' ')
        newword_list = []
        prev_word = ''
   
        for word in word_list:
            word=word.strip()
            ### get the root form of the word
            if word in self.stop_words:
                continue
            
            if word in ["person_name","org_name"]:
                newword_list.append(word)
                continue

            if word in self.negation_words:
                prev_word = word
                continue
                
            ### LEMMATIZE
            if word not in self.dont_lemmat:
                word= self.lemmatizer.lemmatize(word)

                
            if prev_word in self.negation_words:                
                word = 'not_' + word                        
                newword_list.append(word)                        
                prev_word = ''
            else:
                newword_list.append(word)
                
        line = ' '.join(newword_list)

        
        line = re.sub(r'\W', ' ', str(line))
        line = re.sub(r'\d', ' ', line)
        
        line = re.sub(r'br[\s$]', ' ', line)
        line = re.sub(r'\s+[a-z][\s$]', ' ',line)
        line = re.sub(r'b\s+', '', line)
        line = re.sub(r'\s+', ' ', line)
        
        return line
    
    def tokenize_n_gram(self,sentences,n):
        tokens_arr=self.tokenize_lines(sentences)
        ret=[]
        for arr in tokens_arr:
            arr_gram=[]
            pointer=0
            cnt=0
            done=False
            lth=len(arr)
            while done==False:
                if (pointer+n) > lth:
                    done=True
                    break
                arr_gram.append(arr[pointer+cnt])
                cnt+=1
                if cnt == n:
                    cnt=0
                    pointer+=1
            if len(arr_gram) >0:
                ret.append(arr_gram)
        return ret
                
    def tokenize_train_data(self,train_qn_target_lst):
        arr_tokens=[]
        arr_targets=[]
        for (qn,target) in train_qn_target_lst:
            tokens=self.tokenize_line(qn)
            if len(tokens) > 0:
                arr_tokens.append(tokens)
                arr_targets.append(int(target))
        return arr_tokens,arr_targets

