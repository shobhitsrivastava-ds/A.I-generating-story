from flask import Flask,render_template, url_for , redirect, flash, request,send_from_directory

import numpy as np
from PIL import Image
from keras.preprocessing import image

import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import tensorflow.keras.utils as ku
#from tensorflow.keras.np_utils import probas_to_classes

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
#db=SQLAlchemy(app)

#from model import User,Post

#//////////////////////////////////////////////////////////

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'

""" NEW ADDED TEST"""###############################################################################
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
def create_model(predictors, label, max_sequence_len, total_words, epochs):
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
    model.fit(predictors, label, epochs=epochs, verbose=1)
    return(model)

# Function for generating text
def generate_text(seed_text,next_w,max_seq,model):
    for j in range(next_w):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],maxlen=max_seq-1,padding="pre")
        predicted= model.predict_classes(token_list,verbose=1)
        #y_prob = model.predict(token_list) 
        #predicted = y_prob.argmax(axis=-1)
        print(predicted)
        out_word=""
        for word,index in tokenizer.word_index.items():
            if index==predicted:
                out_word = word
                break
        seed_text +=" "+out_word
    return(seed_text)

####################################################################################################
# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['file']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            data= read_file(full_name)
            wds= str(request.form.get("words"))
            n_wds= int(request.form.get("n_words"))
            ep= int(request.form.get("epochs"))
            X, Y , max_len,total_w= data_preperation(data)
            model = create_model(X,Y,max_len,total_w, ep)
            #to_predict_list = request.form.to_dict()
        
            text = generate_text(wds, n_wds, max_len, model)
            return render_template('predict.html', story= text)
        except:
            flash("Upload & Fill the form correctly & then try again", "danger")      
            return redirect(url_for("gen_story"))
        #except :
         #   flash("Please select the image first !!", "success")      
          #  return redirect(url_for("clean"))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/gen_story")
def gen_story():
    return render_template("index.html")

if __name__ == "__main__":
	app.run(debug=True)
