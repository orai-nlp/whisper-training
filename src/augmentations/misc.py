from src.augmentations.augmentation import Augmentation
import src.augmentations.utils as augmentation_utils
import shutil
import librosa
import soundfile as sf


class CopyAugmentation(Augmentation):

	def __init__(self):
		pass

	def apply(self, infile, outfile):
		shutil.copy(infile, outfile)
		return "Copy"


class SpeedAugmentation(Augmentation):
	'''
	Time stretching is the process of changing the speed or duration of an audio signal
	without affecting its pitch.
	'''

	def __init__(self, lower_bound=0.8, upper_bound=2.0):
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

	def apply(self, infile, outfile):
		slower = self.random.uniform(self.lower_bound, 1.0)
		faster = self.random.uniform(1.0, self.upper_bound)
		speed = self.random.choice([slower, 1.0, faster])
		command = f"sox -q -t wav {infile} -t wav {outfile} tempo -s {speed}"
		try:
			augmentation_utils.run_command(command)
			return command
		except:
			print("WARNING! Error processing file. Copying input file to output file")
			shutil.copy(infile, outfile)


class PitchAugmentation(Augmentation):
	'''
	Pitch scaling is the process of changing the pitch without affecting the speed.
	'''

	def __init__(self, lower_bound=-300, upper_bound=300):
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

	def apply(self, infile, outfile):
		pitch = self.random.randint(self.lower_bound, self.upper_bound)
		command = f"sox -q -t wav {infile} -t wav {outfile} pitch {pitch}"
		try:
			augmentation_utils.run_command(command)
			return command
		except:
			print("WARNING! Error processing file. Copying input file to output file")
			shutil.copy(infile, outfile)


class VolumeAugmentation(Augmentation):

	def __init__(self, scale_low=0.125, scale_high=2.0):
		self.scale_low = scale_low
		self.scale_high = scale_high

	def apply(self, infile, outfile):
		volume = self.random.uniform(self.scale_low, self.scale_high)
		command = f"sox -q --vol {volume} -t wav {infile} -t wav {outfile}"
		try:
			augmentation_utils.run_command(command)
			return command
		except:
			print("WARNING! Error processing file. Copying input file to output file")
			shutil.copy(infile, outfile)


class Convert2Wav(Augmentation):

	def __init__(self):
		pass

	def apply(self, infile, outfile):
		command = f"ffmpeg -y -i {infile} -f wav {outfile}"
		try:
			augmentation_utils.run_command(command)
			return command
		except:
			print("WARNING! Error processing file. Copying input file to output file")
			shutil.copy(infile, outfile)


class NormalizeAudio(Augmentation):

	def __init__(self, sampling_rate=16000):
		self.sampling_rate = sampling_rate

	def apply(self, infile, outfile):
		command = f"sox -q -t wav {infile} -r {self.sampling_rate} -c 1 -t wav {outfile}"
		try:
			augmentation_utils.run_command(command)
			return command
		except:
			print("WARNING! Error processing file. Copying input file to output file")
			shutil.copy(infile, outfile)


class MP3ToWAV(Augmentation):
	def apply(self, infile, outfile):
		command = f"ffmpeg -y -i {infile} {outfile}"
		augmentation_utils.run_command(command)
		return command
