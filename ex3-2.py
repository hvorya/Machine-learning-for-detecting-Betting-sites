from nltk.corpus import  stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
import os, sys
import nltk
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from sklearn.svm import OneClassSVM
###############################################################clean webpage
def cleanme(html):
    soup = BeautifulSoup(html, 'lxml') 
    #print(soup.prettify())
    for tag in soup():        
        if tag=="a":
            del tag['href']
        elif tag=="nav" or tag=="table" or tag=="base":
            del tag
        elif tag=="img":
                del tag['src'],tag['heghit'],tag['width']
        elif tag=="div":
            del tag["style"],tag["type"],tag["class"],tag["id"]
        elif tag=="link":
            del tag["rel"], tag["href"],tag["type"]           
        elif tag=="meta":
            del tag["charset"], tag["property"]    
        for attribute in ['lang','language','onmouseover','onmouseout','script','style','font',
    'dir','face','size','color','style','width','height','hspace',
    'border','valign','align','background','bgcolor','link','vlink',
    'alink','cellpadding','cellspacing','id','href','src','type','container','center']:
                    del tag[attribute]                             
    for script in  soup.find_all("script","style","a","img","svg"): 
                    script.extract()   
    text = soup.findAll(text=True)
    return text
############################################################### Reading data and preprocessing
def pre_process(url_path):
        #nltk.download('stopwords')
        #nltk.download('wordnet')
        #stop_words = set(stopwords.words('english'))
        stem=WordNetLemmatizer()
        final_text=""
        try:
                    webpage = requests.get(url_path.strip())        
                    html_code = webpage.content                
        except Exception as e:
                       print(e)
        try:                  
                    web_texts = cleanme(html_code)  # invoking cleanme function
                    text_from_html=' '.join(web_texts)
                    text=re.sub(r'\b\w\b', ' ', text_from_html) # removing single characters
                    clean1 = re.compile(r'<.*?>') # removing tags# removing tags
                    clean2=re.compile(r'(\W|\d)+')  # removing space and digits
                    text=re.sub(clean1, ' ', text)
                    text=re.sub(clean2,' ',text)
                    text= re.sub(r'\s+', ' ', text, flags=re.I)
                    text=text.lower()
                    text=text.split()                     
                    text= [stem.lemmatize(word) for word in text]
                    f_text=[word for word in text if len(word)>=3 if word not in set(["action","border","content","color","FALSE","font","display","http","height","icon","image","img","jpg","left","margin","medium","mobile","name","none","null","disabled","padding","png","gbp","right","rule","size","text","title","top","TRUE","url","type","var","width","line","position","center","align","background","bottom","src","data","item","label","block","file","logo","link","webkit","version","important","hide","generated","class","hover","static","end","body","div","flex","rem","nav","lang","index","header","footer","href","section","radius","justify","else","ssl","svg","enabled","error","active","Opacity","format","inline","style","filter","module","script","opacity","transform","translation","weight","key","max","min","online","child","app","code","col","detail","device","family","site","default","HEAD","get","float","menu","first","main","value","time","animation","btn","component","container","hidden","tag","come","provider","auto","rgba"])]
                    keyword_test=" ".join(f_text)
                    #print(keyword_test)
        except Exception as e:
                        print(e)
        return keyword_test               

############################################################## bag of words model =the output are two files.
def bag_of_words(keyword_train,keyword_test):
          X=keyword_train
          tf = TfidfVectorizer(max_features=200, stop_words=stopwords.words('english'))          
          y = tf.fit_transform(keyword_test) 
          rows=tf.get_feature_names()
          Y=pd.DataFrame(y.T.todense(),index=rows)
          model = OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
          train_vectors=model.fit(X) 
          test = model.predict(y)
          rows=tf.get_feature_names()
          BOW_dataset=pd.DataFrame(y.T.todense(),index=rows)
          return test
##############################################################main
def main():

        if (len(sys.argv) != 2):
              print ('Please specify the url path')

        else:
                url_path = str(sys.argv[1])
                npz_path=os.path.dirname(str(sys.argv[0]))
                keyword_train=scipy.sparse.load_npz(os.path.join(npz_path , "keywords.npz"))           
                keyword_test=pre_process(url_path)
                print("salam")
                prediction=bag_of_words(keyword_train,[keyword_test])
                if prediction==1:
                    print('OUTPUT1: Gambling site')
                else:
                    print('Non-Gambling site')
if __name__=='__main__':
    main()