from transformers import (WhisperProcessor,
	WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer)
import evaluate
import numpy as np
import os
from . import settings
from .dataset import load_multiple_datasets, normalize_text, DataCollatorSpeechSeq2SeqWithPadding
from . import augmentations as aug


def train(config):
	processor = WhisperProcessor.from_pretrained(
		config["base_model"],
		language=settings.LANGUAGE_MAP[config["language"]],
		task="transcribe",
		cache_dir=config["cache_dir"]
	)

	if config["augment"]:
		augmentations = aug.AudioAugmentations([
		aug.NormalizeAudio(sampling_rate=16000),
		aug.Choose([
			aug.CopyAugmentation(),
			aug.PitchAugmentation(lower_bound=-300, upper_bound=300),
		]),
		aug.SpeedAugmentation(lower_bound=0.8, upper_bound=2.0),
		aug.VolumeAugmentation(scale_low=0.125, scale_high=2.0),
		aug.NormalizeAudio(sampling_rate=16000)], seed=2024, debug=False)
	else:
		augmentations = aug.AudioAugmentations([])


	print("Loading training datasets...")
	train_dataset = load_multiple_datasets(config, config["train_data"], processor,
		shuffle=True, audio_augmentations=augmentations)
	print("[OK]")

	print("Loading validation datasets...")
	valid_dataset = load_multiple_datasets(config, config["valid_data"], processor)
	print("[OK]")

	data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

	metric = evaluate.load("wer")

	def compute_metrics(pred):
		pred_ids = pred.predictions
		label_ids = pred.label_ids

		# replace -100 with the pad_token_id
		label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

		# we do not want to group tokens when computing the metrics
		pred_str = [normalize_text(x, config["language"]) for x in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
		label_str = [normalize_text(x, config["language"]) for x in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

		wer = 100 * metric.compute(predictions=pred_str, references=label_str)

		return {"wer": wer}


	model = WhisperForConditionalGeneration.from_pretrained(config["base_model"], cache_dir=config["cache_dir"])
	model.config.forced_decoder_ids = None
	model.config.suppress_tokens = []
	model.config.apply_spec_augment = True
	model.generation_config.language = config["language"]
	model.config.dropout = 0.1


	training_args = Seq2SeqTrainingArguments(
		output_dir=config["output_model_dir"],  # change to a repo name of your choice
		per_device_train_batch_size=config["batch_size"],
		gradient_accumulation_steps=config["gradient_accumulation"],  # increase by 2x for every 2x decrease in batch size
		learning_rate=1e-5,
		warmup_ratio=0.1,
		num_train_epochs=config["epochs"],
		gradient_checkpointing=False,
		fp16=True,
		evaluation_strategy="steps",
		per_device_eval_batch_size=config["eval_batch_size"],
		predict_with_generate=True,
		save_steps=config["save_steps"],
		eval_steps=config["eval_steps"],
		logging_steps=25,
		report_to=["tensorboard"],
		push_to_hub=False,
		dataloader_num_workers=config["dataloader_num_workers"],
		remove_unused_columns=False,
	)

	trainer = Seq2SeqTrainer(
		args=training_args,
		model=model,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		data_collator=data_collator,
		compute_metrics=compute_metrics
	)

	try:
		trainer.train(resume_from_checkpoint=config["resume_from"] if "resume_from" in config else False)
	except ValueError:
		print("No checkpoint found. Training from scratch")
		trainer.train()
