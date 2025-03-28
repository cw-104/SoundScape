import subprocess
import os
import sys

class CladModel:
    def __init__(self):
        self.venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../CLAD/venv"))
        self.clad_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../CLAD/run_clad.py"))
        print(f"CLAD Virtual Env Path: {self.venv_path}")
        print(f"CLAD Script Path: {self.clad_script}")

    def predict(self, input_audio_path):
        try:
            venv_python = os.path.join(self.venv_path, 'bin', 'python')
            pythonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
            cmd = f'PYTHONPATH={pythonpath} {venv_python} {self.clad_script} {input_audio_path}'
            print(f"Executing CLAD subprocess: {cmd}")

            # ✅ Run subprocess and capture both stdout and stderr
            result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)

            print("CLAD STDOUT:", result.stdout)
            print("CLAD STDERR:", result.stderr)

            if result.returncode != 0:
                print("CLAD subprocess error detected.")
                return None

            return result.stdout
        except Exception as e:
            print(f"CLAD subprocess failed: {e}")
            return None
