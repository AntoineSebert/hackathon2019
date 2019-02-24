"""This module generates stories from sample using deep learning."""

import argparse

import json
from html.parser import HTMLParser
from html.entities import name2codepoint

import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text

class datasets_group:
	x_train = {}
	y_train = {}
	x_test = {}
	y_test = {}
	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

class article:
	content = ""
	def __init__(self, html_content):
		self.content = html_content

class MyHTMLParser(HTMLParser):
	def __init__(self):
		HTMLParser.__init__(self)
		self.recording = 0
		self.data = []
	def handle_starttag(self, tag, attrs):
		for attr in attrs:
			print("     attr:", attr)
		if tag == 'p':
			print("Start tag:", tag)
			return
	def handle_endtag(self, tag):
		if tag == 'p':
			print("End tag  :", tag)
			return
	def handle_data(self, data):
		print("Data     :", data)

def get_input_file():
	"""Get the filepath from the command line."""
	parser = argparse.ArgumentParser(description = 'Generate texts from text datatset.')
	parser.add_argument(
		"-f",
		"--file",
		nargs = '?',
		default = 'dataset/medium-articles/train.json',
		type = argparse.FileType('r'),
		dest = "file",
		help = "use dataset in FILE"
	)
	args = parser.parse_args()

	return args.file

def load_data(file):
	"""Extract the data from the file and return it as a list of objects."""
	articles = []
	parser = MyHTMLParser()

	for line in iter(lambda: file.readline(), ''):
		article_as_json = json.loads(line)
		parser.feed(article_as_json["content"])
		print(parser.data)
		print()
		#articles.append(article())

	return articles

def create_datasets(raw_data, top_words, max_review_length):
	"""Create the train and test datasets, each entry being limited in length."""

	# to replace
	(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = top_words)
	# iterate through list of articles and get their content
	#x_train = text.text_to_word_sequence(x_train)
	#x_test = text.text_to_word_sequence(x_test)
	x_train = sequence.pad_sequences(x_train, maxlen = max_review_length)
	x_test = sequence.pad_sequences(x_test, maxlen = max_review_length)

	# https://keras.io/preprocessing/text/

	return datasets_group(x_train, y_train, x_test, y_test)

def create_model(top_words, max_review_length, embedding_size):
	"""Create a model with the given parameters."""
	model = Sequential()
	model.add(Embedding(top_words, embedding_size, input_length = max_review_length))
	#model.add(Dropout(0.2))
	model.add(LSTM(100))
	#model.add(Dropout(0.2))
	model.add(Dense(1, activation = 'sigmoid'))

	return model

def create_neural_network(top_words, max_review_length, embedding_size, loss, optimizer, metrics):
	"""Create a neural network that takes a dataset of texts as input and generates texts based on the dataset."""
	model = create_model(top_words, max_review_length, embedding_size)
	model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])

	return model

def train_neural_network(model, datasets, epochs, batch_size):
	"""Train the neural network using the provided dataset."""
	model.fit(
		datasets.x_train,
		datasets.y_train,
		validation_data = (datasets.x_test, datasets.y_test),
		epochs = epochs,
		batch_size = batch_size
	)

def generate_text(model, datasets):
	"""Generate new text."""
	scores = model.evaluate(datasets.x_test, datasets.y_test, verbose = 0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return article("")

def main():
	"""Script entry point"""
	print('hello world')

	file = get_input_file()

	raw_data = load_data(file)

	top_words = 5000
	max_review_length = 500
	datasets = create_datasets(raw_data, top_words, max_review_length)

	np.random.seed(1337)
	model = create_neural_network(
		top_words,
		max_review_length,
		32,
		'binary_crossentropy',
		'adam',
		'accuracy'
	)

	#train_neural_network(model, datasets, 3, 64)

	#generate_text(model, datasets)

if __name__ == "__main__":
	main()
