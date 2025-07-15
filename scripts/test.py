import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, root_dir)
import argparse
import yaml
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src.test import run_test
import src.augmentations as aug

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test Whisper model on different tests.')
	parser.add_argument('config', help='Experiment configuration file')
	parser.add_argument('model', help='Path to Whisper models, can be local or from the hub.')
	parser.add_argument("results_dir", help="Directory to save results")
	parser.add_argument('--device', default="cuda", help="Device to run inference on.")
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument("--skip", default=[], nargs="+", help="Test sets to skip")
	parser.add_argument("--add_noise", action="store_true", help="Test model robustness by adding noise")
	parser.add_argument("--add_noise_only", action="store_true", help="Test model robustness by adding noise. Skips normal test")
	
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	results_dir = args.results_dir

	processor = WhisperProcessor.from_pretrained(args.model, cache_dir=config["cache_dir"])
	model = WhisperForConditionalGeneration.from_pretrained(args.model, cache_dir=config["cache_dir"])
	model.to(args.device)

	for test_name, test_data in config["test_data"].items():
		print(f"TEST: {test_name}")
		if test_name in args.skip:
			print("Skipping test set")
			continue

		if args.add_noise_only:
			augmentation_setups = []
		else:
			augmentation_setups = [(aug.AudioAugmentations([]), "normal")] # Default is to run normal test

		# Noise benchmarking
		if args.add_noise or args.add_noise_only:
			augmentations = aug.AudioAugmentations([
				aug.NormalizeAudio(sampling_rate=16000),
				aug.VolumeAugmentation(scale_low=0.125, scale_high=2.0),			
				aug.NormalizeAudio(sampling_rate=16000)
			], seed=2024)
			augmentation_setups.append((augmentations, f"{noise_type}_snr{snr}"))

		for i, setup in enumerate(augmentation_setups):
			if not os.path.exists(results_dir):
				os.makedirs(results_dir)
			if setup[1] != "normal":
				print(f"[SETUP]: {setup[1]}")
				name_hash = f"{test_name}_{setup[1]}"
			else:
				name_hash = f"{test_name}"
			results_filepath = os.path.join(results_dir, f"{name_hash}.out")

			augmentations = setup[0]
			run_test(config, test_name, test_data, config["language"], augmentations,
				processor, model, results_filepath, args.batch_size, args.device)
