import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, root_dir)
import argparse
from src.server import TranscriptionServer


def main():
    parser = argparse.ArgumentParser(description="Starts a server with all the Whisper models")
    parser.add_argument('language', help="Language")
    parser.add_argument('model', help="Path to the model")
    parser.add_argument('--host', type=str, default="127.0.0.1", help="Server host name")
    parser.add_argument('--port', type=str, default=5000, help="Server port")
    args = parser.parse_args()

    transcription_server = TranscriptionServer(args.language, args.model)
    transcription_server.run(debug=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
