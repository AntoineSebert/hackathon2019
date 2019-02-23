"""This module generates stories from sample using deep learning."""

import argparse

def get_input_file():
	"""Get the filepath from the command line."""
	parser = argparse.ArgumentParser(description = 'Generate texts from text datatset.')
	parser.add_argument(
		"-f",
		"--file",
		nargs = '?',
		default = 'dataset/medium-articles/train.json',
		type = argparse.FileType('r'),
		required = True,
		dest = "file",
		help = "use dataset in FILE"
	)
	args = parser.parse_args()
	return args.file

def load_data(file):
	"""Extract the data from the file and return it as a list of objects."""
	"""
	for line in iter(lambda: file.readline(), ''):
		print(line)
	"""
	return []

def create_neural_network():
	"""Create a neural network that takes a dataset of texts as input and generates texts based on the dataset."""
	import numpy as np
	from keras.datasets import imdb
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	from keras.layers.embeddings import Embedding
	from keras.preprocessing import sequence

	np.random.seed(1337)
	top_words = 5000
	(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
	max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	def create_model(top_words, max_review_length, embedding_size):
		model = Sequential()
		model.add(Embedding(top_words, embedding_size, input_length=max_review_length))
		#model.add(Dropout(0.2))
		model.add(LSTM(100))
		#model.add(Dropout(0.2))
		model.add(Dense(1, activation='sigmoid'))
		return model

	embedding_size = 32

	loss = 'binary_crossentropy'
	optimizer = 'adam'
	metrics = 'accuracy'

	epochs = 3
	batch_size = 64


	model = create_model(top_words, max_review_length, embedding_size)
	model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return {}

def train_neural_network(neural_network):
	"""Train the neural network using the provided dataset."""

def generate_text(neural_network):
	"""Generate new text."""
	return {}

def main():
	"""Script entry point"""
	print('hello world')

	file = get_input_file()
	print(file)
	dataset = load_data(file)
	neural_network = create_neural_network()
	train_neural_network(neural_network)
	generate_text(neural_network)

if __name__ == "__main__":
	main()
