import re
import evaluate
import torch
import os
import yaml
from functools import partial
from src.dataset import load_test_dataset, save_sample, normalize_text
import src.settings
import glob


def clean_text(text, test_name, language):
	text = normalize_text(text, language)
	return text

def predict_fn(batch, processor, model, test_name, language, device):
	audio = batch["audio"] 
	assert audio[0]["sampling_rate"] == 16000

	input_features = processor([x["array"] for x in audio], sampling_rate=audio[0]["sampling_rate"], return_tensors="pt").input_features
	batch["reference"] = [clean_text(x, test_name, language) for x in batch["text"]]

	with torch.no_grad():
		task = "transcribe"
		predicted_ids = model.generate(input_features.to(device), task=task, language=language)
	transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
	batch["prediction"] = [clean_text(x, test_name, language) for x in transcription]

	return batch

def run_test(config, test_name, test_data, language, augmentations, processor, model, results_filepath, batch_size, device):
	# Load test
	test = load_test_dataset(config, test_data, augmentations)

	# Do inference
	fn = partial(predict_fn, processor=processor, model=model, test_name=test_name, language=language, device=device)
	result = test.map(fn, batched=True, batch_size=batch_size, keep_in_memory=True)

	# Compute WER
	wer_metric = evaluate.load("wer")
	wer_score = 100 * wer_metric.compute(references=result["reference"], predictions=result["prediction"])
	print(f"WER: {wer_score:.2f}")

	# Save results
	with open(results_filepath, "w", encoding="utf-8") as f:
		f.write(f"Final result: {round(wer_score, 2)} WER\n\n")
		for i, (pred, ref) in enumerate(zip(result["prediction"], result["reference"])):
			f.write(f"{i}\n")
			f.write(f"{ref}\n")
			f.write(f"{pred}\n")
			wer_score = 100 * wer_metric.compute(predictions=[pred], references=[ref])
			f.write(f"{wer_score:.2f}\n\n")
	print(f"Results in: {results_filepath}")
