from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,LSTM,Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import pandas as pd
import numpy as np
import argparse
import os

# Taking the file path as input
ap =argparse.ArgumentParser()
ap.add_argument("-t","--text_file",default = os.path.join(os.getcwd(),"sample.text"),help ="Path to the text file")
ap.add_argument("-w","--words",type=str,help="Starting word to test the model")
ap.add_argument("-n","--number",type=int,help="numbers of words to generate")
ap.add_argument("-e","--epochs",type=int,help="number of epochs to use")
var= vars(ap.parse_args())

# Function for readig the file
def read_file(path):
	print("Inside the read file")
	with open(path,"r") as f:
		data= f.read()
	return(data)
# Function for the preperation of the data
tokenizer = Tokenizer()
def data_preperation(data):
	corpus =data.split(".")
	tokenizer.fit_on_texts(corpus)
	total_words = len(tokenizer.word_index)+1
	input_seq=[]
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1,len(token_list)):
			n_gram_seq = token_list[:i+1]
			input_seq.append(n_gram_seq)
	max_seq_len = max([len(x) for x in input_seq])
	input_sequence = np.array(pad_sequences(input_seq,maxlen=max_seq_len,padding="pre"))
	predictors,labels = input_sequence[:,:-1],input_sequence[:,-1]
	label = ku.to_categorical(labels,num_classes =total_words)
	print(total_words)
	print(predictors.shape)
	print(label)
	return(predictors,label,max_seq_len,total_words)

# Function for creating the model
def create_model(predictors, label, max_sequence_len, total_words):
	input_len = max_sequence_len-1	
	model = Sequential()
        model.add(Embedding(total_words, 10, input_length=input_len))
        model.add(LSTM(150))
        model.add(Dropout(0.1))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(predictors.shape)
	print(total_words)
	print(label.shape)
        model.fit(predictors, label, epochs=var["epochs"], verbose=1)
	return(model)
"""def create_model(pred,labels,max_seq,total_words):
	input_len= max_seq-1
	model = Sequential()
	model.add(Embedding(total_words,10,input_length =input_len))
	model.add(LSTM(150))
	model.add(Dense(total_words,activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer = "adam")
	print(pred.shape)
	print(total_words)
	model.fit(pred,labels,epochs=159,verbose=1)
	return(model)"""


# Function for generating text
def generate_text(seed_text,next_w,max_seq,model):
	for j in range(next_w):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list],maxlen=max_seq-1,padding="pre")
		predicted= model.predict_classes(token_list,verbose=1)
		out_word=""
		for word,index in tokenizer.word_index.items():
			if index==predicted:
				out_word = word
				break
		seed_text +=" "+out_word
	return(seed_text)

			
data= read_file(var["text_file"])

X, Y , max_len,total_w= data_preperation(data)
model = create_model(X,Y,max_len,total_w)
text = generate_text(var["words"],var["number"],max_len,model)
print(text)
	









































	

