from flask import Flask, request, jsonify
import json
import librosa
import io
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000

class ServerWhisperModel:

	def __init__(self, language, model, device="cuda"):
		self.language = language
		self.device = device
		self._load_model(model)

	def _load_model(self, model):
		print(f"Loading Whisper model...")
		self.processor = WhisperProcessor.from_pretrained(model)
		self.model = WhisperForConditionalGeneration.from_pretrained(model)
		self.model.to(self.device)
		print(f"[OK]")

	def run(self, audios):
		print(f"Trancribing batch on Whisper model...")
		audio_arrays = []
		for audio in audios:
			audio_bytes = audio.read()
			audio_chunk, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
			assert SAMPLE_RATE == sr
			audio_arrays.append(audio_chunk)

		input_features = self.processor(audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
		with torch.no_grad():
			task = "transcribe"
			predicted_ids = self.model.generate(input_features.to(self.device), task=task, language=self.language)
		transcriptions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
		return transcriptions


class TranscriptionServer:

	def __init__(self, language, model, hf_token=None, device="cuda"):
		self.app = Flask(__name__)

		# Load Whisper model
		self.model = ServerWhisperModel(language, model, device=device)

		# Register the routes
		self.register_routes()

	def register_routes(self):

		@self.app.route('/transcribe', methods=['POST'])
		def transcribe():
			if not request.form.get("batch_size"):
				return jsonify({'error': 'No "batch_size" field provided'}), 400
			audios = [request.files[f'audio_{i}'] for i in range(int(request.form["batch_size"]))]
			transcriptions = self.model.run(audios)
			return jsonify({'transcriptions': transcriptions})

	def run(self, **kwargs):
		self.app.run(**kwargs)
