import librosa
import os
import re
from collections import defaultdict
import subprocess
import tempfile
import soundfile as sf
from . import settings


def run_command(command):
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	output, error = process.communicate()
	if process.returncode != 0:
		print(command)
		print(error.decode("utf-8"))
		raise Exception("Error processing file")


def read_audiofile(filepath, duration=None, offset=0.0):
	array, sampling_rate = librosa.load(filepath, sr=None, mono=True, duration=duration, offset=offset)
	return array, sampling_rate


class File:

	def __init__(self, fileid, filepath):
		self.fileid = fileid
		self.filepath = filepath

		#audio_array, sampling_rate = read_audiofile(filepath)
		#self.audio = filepath
		#self.sampling_rate = sampling_rate

	def get(self, start_time, end_time):
		if not self.filepath.endswith("|"):
			audio_array, sampling_rate = read_audiofile(self.filepath, offset=start_time, duration=end_time-start_time)
		else:
			temp_path = f"/tmp/tmp{next(tempfile._get_candidate_names())}.wav"
			command = re.sub("ffmpeg ", "ffmpeg -y ", re.sub(r" \- ", f" {temp_path} ", self.filepath[:-1])).strip()
			run_command(command) # Run piped audio and convert to WAV
			audio_array, sampling_rate = read_audiofile(temp_path, offset=start_time, duration=end_time-start_time)
			if os.path.isfile(temp_path):
				os.remove(temp_path)

		return audio_array, sampling_rate


class Segment:

	def __init__(self, uttid, file, start, end, text, prompt):
		self.uttid = uttid
		self.file = file
		self.start = float(start)
		self.end = float(end)
		self.text = text
		self.prompt = prompt
		self.duration = self.end - self.start

	@property
	def audio(self):
		return self.file.get(self.start, self.end)


def parse_wav_scp_file(filepath):
	files = dict()
	with open(filepath, "r", encoding="utf-8") as f:
		for line in f:
			fileid, *filepath = line.strip().split()
			filepath = " ".join(filepath)
			filepath = re.sub("ffmpeg", settings.ffmpeg_path, filepath)
			files[fileid] = File(fileid, filepath)
	return files


def parse_text_file(filepath):
	texts = dict()
	with open(filepath, "r", encoding="utf-8") as f:
		for line in f:
			uttid, *text = line.strip().split()
			text = " ".join(text)
			texts[uttid] = text
	return texts


def parse_segments_file(filepath, files, texts, prompts):
	segments = dict()
	with open(filepath, "r", encoding="utf-8") as f:
		for line in f:
			uttid, fileid, start, end = line.strip().split()
			start = float(start)
			end = float(end)
			segments[uttid] = Segment(uttid, files[fileid], start, end, texts[uttid], prompts[uttid])
	return segments


def load_kaldi_test(directory):
	files = parse_wav_scp_file(os.path.join(directory, "wav.scp"))
	if os.path.isfile(os.path.join(directory, "text")):
		texts = parse_text_file(os.path.join(directory, "text"))	
	else:
		texts = defaultdict(str)
	if os.path.isfile(os.path.join(directory, "prompt")):
		prompts = parse_text_file(os.path.join(directory, "prompt"))
	else:
		prompts = defaultdict(str)
	segments = parse_segments_file(os.path.join(directory, "segments"), files, texts, prompts)

	return segments
