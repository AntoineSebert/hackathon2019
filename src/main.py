"""This module generates stories from sample using deep learning."""

import argparse

def get_input_file():
	parser = argparse.ArgumentParser(description = 'Generate texts from text datatset.')
	parser.add_argument(
		"-f",
		"--file",
		nargs = '?',
		default = '../dataset/booksummaries/booksummaries.txt',
		type = argparse.FileType('r'),
		required = True,
		dest = "file",
		help = "use dataset in FILE"
	)
	args = parser.parse_args()
	return args.file

def main():
	"""Script entry point"""
	print('hello world')

	file = get_input_file()
	#dataset = load_data(file)

if __name__ == "__main__":
	main()
