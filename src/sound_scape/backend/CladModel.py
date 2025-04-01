import subprocess
import os
import sys
from Base_Path import get_path_relative_base

class CladModel:
    def __init__(self, debug_print=False):
        # self.venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../CLAD/venv"))
        self.venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), get_path_relative_base("CLAD/venv")))
        # self.clad_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../CLAD/run_clad.py"))
        self.clad_script = os.path.abspath(os.path.join(os.path.dirname(__file__),get_path_relative_base("CLAD/run_clad.py")))
        self.debug_print = debug_print
        if self.debug_print:
            print(f"CLAD Virtual Env Path: {self.venv_path}")
            print(f"CLAD Script Path: {self.clad_script}")

    def predict(self, input_audio_path, separate_py_env=False):
        try:
            cmd = ""
            if separate_py_env:
                venv_python = os.path.join(self.venv_path, 'bin', 'python')
                pythonpath = os.path.abspath(os.path.join(os.path.dirname(__file__), get_path_relative_base(".")))
                cmd = f'PYTHONPATH={pythonpath} {venv_python} {self.clad_script} {input_audio_path}'
            else:
                cmd = f'python {self.clad_script} {input_audio_path}'
            if self.debug_print:
                print(f"Executing CLAD subprocess: {cmd}")

            # ✅ Run subprocess and capture both stdout and stderr
            result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)

            if self.debug_print:
                print("CLAD STDOUT:", result.stdout)
                print("CLAD STDERR:", result.stderr)

            if result.returncode != 0:
                print("CLAD subprocess error detected.")
                return None

            return result.stdout
        except Exception as e:
            print(f"CLAD subprocess failed: {e}")
            return None
