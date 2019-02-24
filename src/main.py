"""This module generates stories from sample using deep learning."""

import argparse

import json
from html.parser import HTMLParser
from html.entities import name2codepoint

import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM
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

def get_input_file():
	"""Get the filepath from the command line."""
	parser = argparse.ArgumentParser(description = 'Generate texts from text datatset.')
	parser.add_argument(
		"-f",
		"--file",
		nargs = '?',
		default = 'dataset/medium-articles/test.json',
		type = argparse.FileType('r'),
		dest = "file",
		help = "use dataset in FILE"
	)
	args = parser.parse_args()

	return args.file

def load_data(file):
	"""Extract the data from the file and return it as a list of objects."""
	interesting_data = ""

	class LinksParser(HTMLParser):
		def __init__(self):
			HTMLParser.__init__(self)
			self.recording = 0
			self.data = []
		def handle_starttag(self, tag, attrs):
			for (key, value) in attrs:
				if key == 'class' and 'graf--p' in value:
					self.recording = 1
				else:
					self.recording = 0
		def handle_data(self, data):
			if self.recording == 1:
				nonlocal interesting_data
				interesting_data += data

	articles = []
	parser = LinksParser()

	limit = 0
	for line in iter(lambda: file.readline(), ''):
		if 100 < limit:
			break
		limit += 1
		article_as_json = json.loads(line)
		parser.feed(article_as_json["content"])
		articles.append(article(interesting_data))
		interesting_data = ""

	return articles

def create_datasets(articles, top_words, max_review_length):
	"""Create the train and test datasets, each entry being limited in length."""

	(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = top_words, test_split=0.2)

	# print elements

	for article in articles:
		np.append(x_train, text.text_to_word_sequence(article.content))
	for article in articles:
		np.append(x_test, text.text_to_word_sequence(article.content))

	# print elements

	x_train = sequence.pad_sequences(x_train, maxlen = max_review_length)
	x_test = sequence.pad_sequences(x_test, maxlen = max_review_length)

	# print elements

	return datasets_group(x_train, y_train, x_test, y_test)

def create_model(top_words, max_review_length, embedding_size):
	"""Create a model with the given parameters."""

	model = Sequential()
	model.add(LSTM(75)) # """, input_shape = (X.shape[1], X.shape[2])"""
	model.add(Dense(units = 1, activation = 'softmax')) # , vocab_size = top_words

	return model

def create_neural_network(top_words, max_review_length, embedding_size, loss, optimizer, metrics):
	"""Create a neural network that takes a dataset of texts as input and generates texts based on the dataset."""
	model = create_model(top_words, max_review_length, embedding_size)
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	return model

def train_neural_network(model, datasets, epochs, batch_size):
	"""Train the neural network using the provided dataset."""
	model.fit(datasets.x_train, (datasets.x_test, datasets.y_test), epochs = 100, verbose = 2)

def generate_text(model, datasets):
	"""Generate new text."""
	scores = model.evaluate(datasets.x_test, datasets.y_test, verbose = 0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	return article("")

def main():
	"""Script entry point"""

	file = get_input_file()

	articles = load_data(file)

	top_words = 5000
	max_review_length = 500
	datasets = create_datasets(articles, top_words, max_review_length)

	np.random.seed(1337)
	model = create_neural_network(
		top_words,
		max_review_length,
		32,
		'binary_crossentropy',
		'adam',
		'accuracy'
	)

	train_neural_network(model, datasets, 3, 64)

	#generate_text(model, datasets)

if __name__ == "__main__":
	main()
