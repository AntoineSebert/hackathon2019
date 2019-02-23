"""This module generates stories from sample using deep learning."""

import argparse
import numpy
import tensorflow
import keras

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
	a = numpy.arange(15).reshape(3, 5)
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
