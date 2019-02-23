import numpy as np
import keras
import tensoflow
#from keras.datasets import imdb
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(1337)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# neural network architecture

def create_model(top_words, max_review_length, embedding_size):
    model = Sequential()
    model.add(Embedding(top_words, embedding_size, input_length=max_review_length))
    #model.add(Dropout(0.2))
    model.add(LSTM(100))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return mod

# fit the architecture

embedding_size = 32

loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = 'accuracy'

epochs = 3
batch_size = 64


model = create_model(top_words, max_review_length, embedding_size)
model.compile(loss=loss, 
              optimizer=optimizer,
              metrics=[metrics])

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=epochs,
          batch_size=batch_size)

#Accuracy Measure

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
