# -*- coding: utf-8 -*-

"""This module generates stories from sample using deep learning."""

import argparse

import json
from html.parser import HTMLParser
from html.entities import name2codepoint

import numpy as np
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

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
		type = argparse.FileType('r', encoding="utf-8"),
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
		if 9 < limit:
			break
		limit += 1
		article_as_json = json.loads(line)
		parser.feed(article_as_json["content"])
		articles.append(article(interesting_data))
		interesting_data = ""

	return articles

def create_datasets(articles, top_words, max_review_length):
	"""Create the train and test datasets, each entry being limited in length."""

	raw_content = ""
	for article in articles:
		raw_content += article.content

	# create tokenizer
	indexer = text.Tokenizer(num_words=max_review_length, filters='"#$%&*+-/<=>@[\\]^_`{|}~\t\n', lower=False, split=' ', char_level=False)

	# prepare tokenizer
	indexer.fit_on_texts(raw_content)

	# break str into words
	words_sequence = text.text_to_word_sequence(raw_content, filters='"#$%&*+-/<=>@[\\]^_`{|}~\t\n', lower=False, split=' ')

	# assign indexes to words
	word_index = indexer.word_index
	print('Found %s unique tokens.' % len(word_index))
	word_indices = dict((c, i) for i, c in enumerate(words_sequence))
	indices_word = dict((i, c) for i, c in enumerate(words_sequence))

	print(words_sequence)

	return datasets_group(words_sequence, len(word_index), words_sequence, {})

def create_model(max_review_length, vocab, loss, optimizer, metrics):
	"""Create a model with the given parameters."""

	model = Sequential()
	model.add(Bidirectional(LSTM(128, input_shape=(max_review_length, vocab))))
	model.add(Dropout(0.2))
	model.add(Dense(units = 32, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	return model

def train_neural_network(model, datasets, epochs):
	"""Train the neural network using the provided dataset."""

	model.fit(datasets.x_train, epochs, verbose = 2, batch_size=128)

def generate_text(max_review_length, vocab, model, datasets, words_sequence):
	"""Generate new text."""
	"""
	def generator(sentence_list, next_word_list, batch_size):
		index = 0
		while True:
			x = np.zeros((batch_size, max_review_length), dtype=np.int32)
			y = np.zeros((batch_size), dtype=np.int32)
			for i in range(batch_size):
				for t, w in enumerate(sentence_list[index]):
					x[i, t, word_indices[w]] = 1
				y[i, word_indices[next_word_list[index]]] = 1
				index = index + 1
				if index == len(sentence_list):
					index = 0
			yield x, y

	model.fit_generator(
		generator(words_sequence, next_words, BATCH_SIZE),
		steps_per_epoch=int(len(words_sequence)/BATCH_SIZE) + 1,
		epochs=100
	)
"""
	def sample(preds, temperature=1.0):
		# helper function to sample an index from a probability array
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)

	generated = ''
	sentence = "Test"
	generated += sentence

	print('----- Generating with seed: "' + sentence + '"')

	for i in range(400):
		x_pred = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
			x_pred[0, t, char_indices[char]] = 1.

		preds = model.predict(x_pred, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]

		generated += next_char
		sentence = sentence[1:] + next_char

		sys.stdout.write(next_char)
		sys.stdout.flush()
		print()

	return ""

def main():
	"""Script entry point"""

	file = get_input_file()

	articles = load_data(file)

	top_words = 5000
	max_review_length = 500
	np.random.seed(1337)
	datasets = create_datasets(articles, top_words, max_review_length)

	model = create_model(
		max_review_length,
		datasets.y_train,
		'sparse_categorical_crossentropy',
		'adam',
		'accuracy'
	)

	train_neural_network(model, datasets, 100)

	generate_text(max_review_length, datasets.y_train, model, datasets, datasets.x_test)

if __name__ == "__main__":
	main()
