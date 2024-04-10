import nltk
nltk.download('punkt')
nltk.download('wordnet')
import json
import time
import pickle
import numpy as np

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()



words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

# opening JSON file, reading data from it, then closing it.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words



# creating a function to make corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
    
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    print(stem_words)
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list




def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:


        pattern_words = word_tags[0] 
        bag_of_words = []
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)
        for word in stem_words:            
            if word in stemmed_pattern_word:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
                bag.append(bag_of_words)

    return np.array(bag)
def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        labels_encoding = list([0]*len(classes))  
        tag = word_tags[1]   # 'greetings'
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(tag_classes, open('classes.pkl','wb'))

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data  , label_data = preprocess_train_data()
print("first BAG OF WORDS encoding: " , bow_data[0])
print("first Label encoding: " , label_data[0])


