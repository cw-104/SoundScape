from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import numpy as np
import soundfile as sf
import subprocess as sp
import os
import shutil
import select
import io
import sys
from typing import Dict, Tuple, Optional, IO


def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()



def separate_file(input_file, output_dir, model='htdemucs', mp3=False, mp3_rate=128, float32=False, int24=False, two_stems=None, trim_silence=False):
    """
    Separate a single audio file using demucs and save it to the specified output directory.

    Args:
        input_file (str): Path to the input audio file.
        output_dir (str): Path to the output directory.
        model (str): Demucs model to use.
        mp3 (bool, optional): Whether to output in MP3 format. Defaults to False.
        mp3_rate (int, optional): MP3 bitrate. Defaults to 320.
        float32 (bool, optional): Whether to use float32 output. Defaults to False.
        int24 (bool, optional): Whether to use int24 output. Defaults to False.
        two_stems (str, optional): Two-stems model to use. Defaults to None.

    Returns:
        str: Full path to the separated file.
    """

    # Construct the demucs command
    cmd = ["python3", "-m", "demucs.separate", "-o", output_dir, "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]

    # Run the demucs command
    p = sp.Popen(cmd + [input_file], stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()

    # Check if the command was successful
    if p.returncode != 0:
        print("Command failed, something went wrong.")
        return None

    # Get the name of the input file without the extension
    filename = os.path.basename(input_file)
    filename_no_ext = os.path.splitext(filename)[0]

    # Move the separated file to the output directory
    separated_file_path = os.path.join(output_dir, filename_no_ext.replace(' ', '') + "_sep.mp3")
    vocals_mp3 = os.path.join(output_dir, model, filename_no_ext, "vocals.mp3")
    vocals_wav = os.path.join(output_dir, model, filename_no_ext, "vocals.wav")

    if os.path.exists(vocals_mp3):
        shutil.move(vocals_mp3, separated_file_path)
    elif os.path.exists(vocals_wav):
        shutil.move(vocals_wav, separated_file_path)
    else:
        raise FileNotFoundError(f"Could not find vocals at {vocals_mp3} or {vocals_wav}")

    if not trim_silence:
        return separated_file_path
    else:
        return audio_trim_silence(separated_file_path, os.path.join(output_dir, filename_no_ext.replace(' ', '') + "_sep_sil.mp3"))

# alias for separate_file function for better name
isolate_file = separate_file

def audio_trim_silence(file, out_file):
    # Load the audio
    audio = AudioSegment.from_file(file)

    # Set more lenient silence detection parameters
    silence_thresh = -43  # Set a lower threshold if needed (e.g., -40)
    min_silence_len = 3000  # Shorten the required silence duration (in ms)

    # Detect non-silent chunks
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Combine non-silent chunks
    if non_silent_ranges:
        non_silent_audio = [audio[start:end] for start, end in non_silent_ranges]
        trimmed_audio = sum(non_silent_audio)
        trimmed_audio.export(out_file, format="mp3")
        return out_file
    else:
        print("File is all silence... not changing")
        return file