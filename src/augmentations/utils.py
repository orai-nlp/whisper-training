import subprocess
import shlex


def run_command(command):
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	output, error = process.communicate()
	if process.returncode != 0:
		print(command)
		print(error.decode("utf-8"))
		raise Exception("Error processing file")

