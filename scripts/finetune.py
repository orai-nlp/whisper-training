import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, root_dir)
import argparse
import random
import yaml
from src.train import train


if __name__ == "__main__":
	random.seed(2023)

	parser = argparse.ArgumentParser(description='Script to finetune Whisper models.')
	parser.add_argument('config', help='Experiment configuration file')
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	train(config)
