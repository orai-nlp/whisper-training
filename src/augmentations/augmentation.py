import random
import librosa
import shutil
import tempfile
import os


class Augmentation:

	def __init__(self):
		pass

	def set_seed(self, random):
		self.random = random

	def apply(self, infile, outfile):
		raise NotImplementedError


class Choose:

	def __init__(self, augmentations=[]):
		self.augmentations = augmentations

	def set_seed(self, random):
		self.random = random
		for aug in self.augmentations:
			aug.set_seed(self.random)

	def apply(self, infile, outfile):
		augmentation = self.random.choice(self.augmentations) # Uniform distribution
		return augmentation.apply(infile, outfile)


class AudioAugmentations:

	def __init__(self, augmentations=[], seed=None, debug=False):
		self.augmentations = augmentations
		self.debug = debug

		if seed is not None:
			self.random = random.Random(seed)
		else:
			self.random = random.Random()

		for aug in self.augmentations:
			aug.set_seed(self.random)

	def set_seed(self, seed):
		self.random = random.Random(seed)
		for aug in self.augmentations:
			aug.set_seed(self.random)

	def apply(self, infile, outfile=None, return_array=False):
		temp_path = f"/tmp/tmp{next(tempfile._get_candidate_names())}.wav"
		shutil.copy(infile, temp_path)

		commands = []
		for aug in self.augmentations:
			temp_path2 = f"/tmp/tmp{next(tempfile._get_candidate_names())}.wav"
			cmd = aug.apply(temp_path, temp_path2)
			shutil.copy(temp_path2, temp_path)
			if os.path.isfile(temp_path2):
				os.remove(temp_path2)
			commands.append(cmd)

		if self.debug:
			for cmd in commands:
				print(cmd)

		if outfile:
			shutil.move(temp_path, outfile)
		if return_array:
			audio_array, sampling_rate = librosa.load(temp_path, sr=None, mono=True)
			if os.path.isfile(temp_path):
				os.remove(temp_path)
			return audio_array, sampling_rate
