from datasets import load_dataset, concatenate_datasets
import gruut
import re
import soundfile as sf
import tempfile
import os
import numpy as np
from torch.utils.data import Dataset
import datasets
import random
import multiprocessing as mp
import tqdm
from functools import partial
import numpy as np
from .kaldi import load_kaldi_test
import math
from collections import defaultdict
import datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

random.seed(2023)

def normalize_text_english(text):
	try:
		result = []
		sents = list(gruut.sentences(text.lower(), lang="en-us"))
		for sent in sents:
			result.append(" ".join([word.text for word in sent]))
		return " ".join(result)
	except:
		print("Gruut error")
		print(text)
		print("-----------")
		return text

def remove_punctuation(text):
	text = re.sub(r"[^\w]", " ", text)
	text = re.sub("_", " ", text)
	text = re.sub(r" +", " ", text)
	return text.strip().upper()

def normalize_text(text, language):
	if language == "en":
		text = normalize_text_english(text)
	text = remove_punctuation(text)
	return text

def prepare_text(sample, language):
	text = sample["text"]
	text = normalize_text(text, language)
	sample["text"] = text.lower()
	return sample

def save_sample(sample, audio_path):
	audio = sample["audio"]
	sf.write(audio_path, audio["array"], audio["sampling_rate"])

def audio_transform(sample, augmentations):
	temp_path = f"/tmp/tmp{next(tempfile._get_candidate_names())}.wav"
	save_sample(sample, temp_path) # Temporarily save audio for data augmentation

	# Audio augmentations
	augmented_array, sampling_rate = augmentations.apply(temp_path, return_array=True)
	sample["audio"]["array"] = augmented_array
	sample["audio"]["sampling_rate"] = sampling_rate

	if os.path.isfile(temp_path):
		os.remove(temp_path)

	return sample

def get_audio_length(sample):
	audio = sample["audio"]
	sample["input_length"] = len(audio["array"])
	return sample

def is_audio_in_length_range(length):
	sampling_rate = 16000
	max_duration_in_seconds = 30.0
	return 0 < length < int(max_duration_in_seconds * sampling_rate)

def prepare_dataset(sample, language, processor, with_prompts=False):
	# load and resample audio data from 48 to 16kHz
	audio = sample["audio"]

	assert audio["sampling_rate"] == 16000

	# compute log-Mel input features from input audio array 
	sample["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

	# encode target text to label ids
	if with_prompts:
		if random.uniform(0,1) <= 0.5:
			sample["labels"] = np.array(processor.tokenizer(" " + sample["text"], sample["prompt"]).input_ids)
		else:
			sample["labels"] = np.array(processor.tokenizer(" " + sample["text"]).input_ids)
	else:
		sample["labels"] = np.array(processor.tokenizer(" " + sample["text"]).input_ids)
	#print(processor.tokenizer.convert_ids_to_tokens(sample["labels"]))

	sample["input_length"] = len(audio["array"])

	return sample

def load_hf_dataset_wrapper(data, config):
	dataset = load_dataset(data["hf_name"], *data["args"], **data["kwargs"], streaming=False, trust_remote_code=True, cache_dir=config["cache_dir"])
	dataset = dataset.cast_column("audio", datasets.features.Audio(16000))
	if data["text_field"] != "text":
		dataset = dataset.rename_column(data["text_field"], "text")
	dataset_features = dataset.features.keys()
	columns_to_keep = {"audio", "text"}
	dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))

	if "multiply" in data:
		print(f"Multiplying dataset by {data['multiply']}")
		dss = [dataset] * data["multiply"]
		dataset = concatenate_datasets(dss)

	if "max_samples" in data:
		if data["max_samples"] <= len(dataset):
			print(f"Dataset has {len(dataset)} samples. Selecting only {data['max_samples']} samples")
			dataset = dataset.select(range(data["max_samples"]))

	return dataset

# Helper function to ensure same random transformation with multiprocess
def audio_transform_random_wrapper(sample, index, augmentations, seeds):
	augmentations.set_seed(seeds[index])
	return audio_transform(sample, augmentations)

def do_augmentations(dataset, augmentations):
	rnd_gnr = random.Random(2024)
	# Set seed for each example
	seeds = [rnd_gnr.randint(0, 1000000000) for x in range(len(dataset))]
	fn = partial(audio_transform_random_wrapper, augmentations=augmentations, seeds=seeds)
	augmented_dataset = dataset.map(fn, num_proc=16, with_indices=True, keep_in_memory=True)
	return augmented_dataset

def load_test_dataset(config, data, augmentations):
	test = load_hf_dataset_wrapper(data, config)
	test = do_augmentations(test, augmentations) # Not on the fly
	return test

class TooLongAudio(Exception):
	pass

class HFDataset(Dataset):
	"""
	Wrapper for HuggingFace dataset. USeful for on the fly transforms.
	On the fly transofrms are essential to save disk space as HF stores
	all the transforms on disk.
	"""
	def __init__(self, dataset, language, processor, shuffle=False, audio_augmentations=None, dump_samples=False):
		self.dataset = dataset
		self.language = language
		self.audio_augmentations = audio_augmentations
		self.dump_samples = dump_samples
		self.processor = processor

		self.ids = list(range(len(self.dataset)))
		if shuffle:
			random.shuffle(self.ids)

	def __len__(self):
		return len(self.dataset)

	def _build_sample(self, sample):
		# These are applied on the fly
		sample = prepare_text(sample, self.language)
		if self.audio_augmentations is not None:
			sample = audio_transform(sample, self.audio_augmentations)

		if not is_audio_in_length_range(len(sample["audio"]["array"])):
			raise TooLongAudio()

		sample = prepare_dataset(sample, self.language, self.processor, with_prompts=False)
		return sample

	def _save_sample(self, sample, idx):
		dump_dir = "dumped"
		if not os.path.exists(dump_dir):
			os.makedirs(dump_dir)
		audio_path = os.path.join(dump_dir, f"audio_{str(idx).zfill(9)}.wav")
		save_sample(sample, audio_path)

	def __getitem__(self, idx):
		real_idx = self.ids[idx]
		while True:
			sample = self.dataset[real_idx]
			try:
				built_sample = self._build_sample(sample)
				if self.dump_samples:
					self._save_sample(built_sample, real_idx)
				return built_sample
			except TooLongAudio:
				# Return another sample randomly
				print("Too long audio, trying with another one")
				real_idx = random.choice(self.ids)

class LocalDataset(Dataset):

	def __init__(self, kaldi_dir, multiply=1, max_samples=-1):
		self.segments = [x for x in load_kaldi_test(kaldi_dir).values()]

		if multiply > 1:
			print(f"  * Multiplying dataset by {multiply}")
			self.segments = self.segments * multiply

		if max_samples > 0:
			print(f"  * Dataset has {len(self.segments)} samples. Selecting only {max_samples} samples")
			self.segments = self.segments[:max_samples]

	def __len__(self):
		return len(self.segments)

	def _build_sample(self, segment):
		audio_array, sampling_rate = segment.audio # This is where the audio file is loaded
		sample = {
			"segment": segment.uttid,
			"text": segment.text,
			"audio": {
				"array": audio_array,
				"sampling_rate": sampling_rate
			}
		}
		return sample

	def __getitem__(self, idx):
		segment = self.segments[idx]
		return self._build_sample(segment)

class ConcatDataset(Dataset):

	def __init__(self, datasets):
		self.segments = []
		for ds in datasets:
			self.segments.extend(ds.segments)

	def __len__(self):
		return len(self.segments)

	def _build_sample(self, segment):
		audio_array, sampling_rate = segment.audio # This is where the audio file is loaded
		sample = {
			"segment": segment.uttid,
			"text": segment.text,
			"audio": {
				"array": audio_array,
				"sampling_rate": sampling_rate
			}
		}
		return sample

	def __getitem__(self, idx):
		segment = self.segments[idx]
		return self._build_sample(segment)

def load_multiple_datasets(config, datasets, processor, shuffle=False, audio_augmentations=None, dump_samples=False):
	loaded_datasets = []
	local_datasets = False
	for dataset_name, data in datasets.items():
		print(f"Loading {dataset_name}...")
		if "kaldi_dir" in data:
			ds = LocalDataset(**data)
			local_datasets = True
		else:
			ds = load_hf_dataset_wrapper(data, config)
		loaded_datasets.append(ds)
		print(f"  ==> {len(ds)} samples")

	if local_datasets:
		concat_dataset = ConcatDataset(loaded_datasets)
	else:
		if len(loaded_datasets) == 1:
			concat_dataset =loaded_datasets[0]
		else:
			cocat_dataset = concatenate_datasets(loaded_datasets)

	final_dataset = HFDataset(concat_dataset, config["language"], processor,
				shuffle=shuffle, audio_augmentations=audio_augmentations,
				dump_samples=dump_samples)
	return final_dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	processor: Any

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# split inputs and labels since they have to be of different lengths and need different padding methods
		# first treat the audio inputs by simply returning torch tensors
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

		# get the tokenized label sequences
		label_features = [{"input_ids": feature["labels"]} for feature in features]
		# pad the labels to max length
		labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		# if bos token is appended in previous tokenization step,
		# cut bos token here as it's append later anyways in shift_tokens_right
		if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
			labels = labels[:, 1:]

		batch["labels"] = labels

		return batch
