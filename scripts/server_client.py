import requests
import argparse


def transcribe_whisper_with_server(audio_segments, host, port):
    url = "http://127.0.0.1:5000/transcribe"

    files = {}
    for i, segment in enumerate(audio_segments):
        files[f"audio_{i}"] = (f"audio_chunk_{i}.wav", segment, "audio/wav")

    data = {"batch_size": i+1}

    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        error = response.json()
        raise Exception(f"Error while transcribing with Whisper server: {error['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts a server with all the Whisper models")
    parser.add_argument('audio', help="Audio path (.wav)")
    parser.add_argument('--host', type=str, default="127.0.0.1", help="Server host name")
    parser.add_argument('--port', type=str, default=5000, help="Server port")
    args = parser.parse_args()

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()
    transcriptions = transcribe_whisper_with_server([audio_bytes], args.host, args.port)
    print(transcriptions)
