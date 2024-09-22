import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings
import random
import csv
import json  # Added for storing preset ratings
from datetime import datetime

from einops import rearrange
import torch
import gradio as gr
import numpy as np  # Added for numerical computations

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen

MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
IS_BATCHED = False  # Since we're running locally and not using batched mode
INTERRUPTING = False

# Define the directory to save generated tracks
GENERATED_TRACKS_DIR = Path("generated_tracks")
GENERATED_TRACKS_DIR.mkdir(exist_ok=True)

# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg logging
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr

# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []
        self.excluded_dirs = {GENERATED_TRACKS_DIR.resolve()}  # Exclude generated_tracks_dir

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        resolved_path = Path(path).resolve()
        if resolved_path.parent in self.excluded_dirs:
            # Do not add files from excluded directories
            return
        self.files.append((time.time(), resolved_path))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def load_model():
    global MODEL
    model_version = 'facebook/musicgen-medium'  # Fixed model
    print("Loading model", model_version)
    if MODEL is None or MODEL.name != model_version:
        # Clear PyTorch CUDA cache and delete model
        if MODEL is not None:
            del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = MusicGen.get_pretrained(model_version, device='cuda' if torch.cuda.is_available() else 'cpu')


def _do_predictions(texts, duration, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("Generating with texts:", texts)
    be = time.time()

    try:
        outputs = MODEL.generate(texts, progress=progress)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])

    outputs = outputs.detach().cpu().float()
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    print("Generation finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_wavs


# Updated the preference options to include 'Tie' and 'Both are bad'
preference_options = ["Generated Music 1", "Generated Music 2", "Tie", "Both are bad"]

# Added a list of generation parameter presets
generation_presets = [
    {'name': 'Preset 1', 'mirostat_eta': 0.1, 'mirostat_tau': [2.2, 2.7, 2.7, 2.7], 'temperature': 1.9, 'cfg_coef': 3.0},
    {'name': 'Preset 2', 'mirostat_eta': 0.1, 'mirostat_tau': [2.4, 2.7, 2.7, 2.7], 'temperature': 1.1, 'cfg_coef': 3.0},
    {'name': 'Preset 3', 'mirostat_eta': 0.1, 'mirostat_tau': [3.8, 2.7, 2.7, 2.7], 'temperature': 1.2, 'cfg_coef': 3.0},
    {'name': 'Preset 4', 'mirostat_eta': 0.1, 'mirostat_tau': [1.2, 1.8, 1.8, 1.8], 'temperature': 1.1, 'cfg_coef': 3.0}
    # Add more presets as needed
]

# Initialize or load ELO ratings
def load_preset_ratings(filename='preset_ratings.json'):
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        # Initialize all presets with a rating of 1500
        return {preset['name']: 1500 for preset in generation_presets}

def save_preset_ratings(preset_ratings, filename='preset_ratings.json'):
    with open(filename, 'w') as f:
        json.dump(preset_ratings, f)

def update_elo_ratings(preset1_name, preset2_name, preference, preset_ratings, K=32):
    R1 = preset_ratings[preset1_name]
    R2 = preset_ratings[preset2_name]

    # Calculate expected scores
    E1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
    E2 = 1 / (1 + 10 ** ((R1 - R2) / 400))

    # Assign actual scores based on user preference
    if preference == "Generated Music 1":
        S1, S2 = 1, 0
    elif preference == "Generated Music 2":
        S1, S2 = 0, 1
    elif preference == "Tie":
        S1, S2 = 0.5, 0.5
    else:  # "Both are bad"
        S1, S2 = 0, 0

    # Update ratings
    preset_ratings[preset1_name] = R1 + K * (S1 - E1)
    preset_ratings[preset2_name] = R2 + K * (S2 - E2)

def compute_sampling_probabilities(preset_ratings, temperature=400):
    ratings = np.array(list(preset_ratings.values()))
    names = list(preset_ratings.keys())

    # Apply softmax function
    exp_ratings = np.exp(ratings / temperature)
    probabilities = exp_ratings / exp_ratings.sum()

    sampling_weights = dict(zip(names, probabilities))
    return sampling_weights

def select_presets_with_probabilities(sampling_weights, num_presets=2):
    preset_names = list(sampling_weights.keys())
    weights = np.array([sampling_weights[name] for name in preset_names])
    # Ensure the weights sum to 1
    weights /= weights.sum()
    # Randomly select presets without replacement based on weights
    selected_names = np.random.choice(preset_names, size=num_presets, replace=False, p=weights)
    # Retrieve the actual preset dictionaries
    selected_presets = [next(preset for preset in generation_presets if preset['name'] == name) for name in selected_names]
    return selected_presets

def predict_full(text, duration, session_state, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    load_model()

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")

    MODEL.set_custom_progress_callback(_progress)

    # Load preset ratings
    preset_ratings = load_preset_ratings()

    # Compute sampling probabilities
    sampling_weights = compute_sampling_probabilities(preset_ratings)

    # Select two presets based on sampling probabilities
    preset1, preset2 = select_presets_with_probabilities(sampling_weights, num_presets=2)

    # Keep the preset descriptions
    preset_description1 = preset1['name']
    preset_description2 = preset2['name']

    # Remove 'name' from the preset dictionaries
    preset1_params = {k: v for k, v in preset1.items() if k != 'name'}
    preset2_params = {k: v for k, v in preset2.items() if k != 'name'}

    # Generate the first track with preset1
    wavs1 = _do_predictions(
        [text], duration, progress=True,
        gradio_progress=progress, **preset1_params)

    # Generate the second track with preset2
    wavs2 = _do_predictions(
        [text], duration, progress=True,
        gradio_progress=progress, **preset2_params)

    # Store the presets and text prompt for later use
    session_state.update({
        'preset1': preset1,
        'preset2': preset2,
        'text': text,
        'duration': duration,
        'has_submitted': False,  # Initialize submission flag
        'selected_preference': None  # Initialize selected preference
    })

    # Return the audio outputs, make preference and optional fields visible, and store session state
    return (
        wavs1[0],
        wavs2[0],
        gr.update(visible=True),  # Show preference Radio
        gr.update(visible=True),  # Show Username Textbox
        gr.update(visible=True),  # Show Comment Textbox
        gr.update(visible=True),  # Show Submit Preference Button
        gr.update(visible=False),  # Hide Confirmation Message
        session_state
    )

def on_submit_preference(preference, username, comment, session_state):
    # Check if the user has already submitted their preference for this generation
    if session_state.get('has_submitted', False):
        # Retrieve the previously selected preference
        selected_preference = session_state.get('selected_preference', 'No preference recorded.')
        # Inform the user that they've already submitted along with their selection
        confirmation = gr.Markdown(f"**You have already submitted your preference for this generation: {selected_preference}. Thank you!**", visible=True)
        # Keep the preset information visible
        preset_description1 = session_state['preset1']['name']
        preset_description2 = session_state['preset2']['name']
        # Disable the preference radio, username, comment fields, and submit button
        disable_preference = gr.update(visible=False)
        disable_username = gr.update(visible=False)
        disable_comment = gr.update(visible=False)
        disable_submit = gr.update(visible=False)
        return (
            preset_description1,
            preset_description2,
            confirmation,
            disable_preference,
            disable_username,
            disable_comment,
            disable_submit,
            session_state  # Return the unchanged session_state
        )
    else:
        # Record the user's choice along with optional fields
        record_user_choice(preference, session_state['preset1'], session_state['preset2'],
                          session_state['text'], session_state['duration'],
                          username, comment)
        # Set the submission flag to True and store the selected preference
        session_state['has_submitted'] = True
        session_state['selected_preference'] = preference

        # Display preset information
        preset_description1 = session_state['preset1']['name']
        preset_description2 = session_state['preset2']['name']

        # Display a thank you message or confirmation including the selected preference
        confirmation = gr.Markdown(f"**Thank you for your feedback! The first preset: {preset_description1}. The second preset: {preset_description2}.**", visible=True)

        # Disable the preference radio, username, comment fields, and submit button to prevent multiple submissions
        disable_preference = gr.update(visible=False)
        disable_username = gr.update(visible=False)
        disable_comment = gr.update(visible=False)
        disable_submit = gr.update(visible=False)

        return (
            preset_description1,
            preset_description2,
            confirmation,
            disable_preference,
            disable_username,
            disable_comment,
            disable_submit,
            session_state  # Return the updated session_state
        )

def record_user_choice(preference, preset1, preset2, text_prompt, duration, username, comment):
    data = {
        'timestamp': datetime.now().isoformat(),
        'preference': preference,
        'text_prompt': text_prompt,
        'duration': duration,
        'preset1_name': preset1['name'],
        'preset1_params': {k: v for k, v in preset1.items() if k != 'name'},
        'preset2_name': preset2['name'],
        'preset2_params': {k: v for k, v in preset2.items() if k != 'name'},
        'username': username if username.strip() else None,
        'comment': comment if comment.strip() else None
    }

    # Define the CSV file path
    csv_file = 'user_preferences.csv'

    # Check if the file exists to write headers
    file_exists = os.path.isfile(csv_file)

    # Write to the CSV file
    with open(csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'timestamp', 'preference', 'text_prompt', 'duration',
            'preset1_name', 'preset1_params',
            'preset2_name', 'preset2_params',
            'username', 'comment'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    # Load current ratings
    preset_ratings = load_preset_ratings()

    # Update ratings
    update_elo_ratings(preset1['name'], preset2['name'], preference, preset_ratings)

    # Save updated ratings
    save_preset_ratings(preset_ratings)

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # MusicGen

            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                text = gr.Text(label="Input Text", interactive=True)
                duration = gr.Slider(minimum=1, maximum=30, value=10, label="Duration (seconds)", interactive=True)
                submit = gr.Button("Generate")
                interrupt_btn = gr.Button("Interrupt").click(fn=interrupt, queue=False)
            with gr.Column():
                audio_output1 = gr.Audio(label="Generated Music 1", type='filepath')
                audio_output2 = gr.Audio(label="Generated Music 2", type='filepath')
                preference = gr.Radio(choices=preference_options,
                                      label="Which track do you prefer?", visible=False)
                username = gr.Textbox(label="Username (optional)", placeholder="Enter your username...", visible=False)
                comment = gr.Textbox(label="Comment (optional)", lines=3, placeholder="Leave your comments here...", visible=False)
                submit_preference = gr.Button("Submit Preference", visible=False)
                preset_info1 = gr.Markdown(label="Preset Used for Track 1", visible=False)
                preset_info2 = gr.Markdown(label="Preset Used for Track 2", visible=False)
                confirmation = gr.Markdown("", visible=False)  # For confirmation messages
                session_state = gr.State({})  # Initialize as an empty dict
        submit.click(
            predict_full,
            inputs=[text, duration, session_state],
            outputs=[
                audio_output1,
                audio_output2,
                preference,
                username,
                comment,
                submit_preference,
                confirmation,
                session_state
            ]
        )
        submit_preference.click(
            on_submit_preference,
            inputs=[preference, username, comment, session_state],
            outputs=[
                preset_info1,
                preset_info2,
                confirmation,
                preference,
                username,
                comment,
                submit_preference,
                session_state
            ]
        )
        gr.Examples(
            fn=predict_full,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    10
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    10
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    10
                ],
                [
                    "A light and cheerful EDM track with syncopated drums, airy pads, and strong emotions",
                    10
                ],
                [
                    "Lofi slow bpm electro chill with organic samples",
                    10
                ],
                [
                    "Punk rock with loud drums and powerful guitar",
                    10
                ],
            ],
            inputs=[text, duration, session_state],
            outputs=[
                audio_output1,
                audio_output2,
                preference,
                username,
                comment,
                submit_preference,
                confirmation,
                session_state
            ],
            cache_examples=False  # Disable caching for dynamic outputs
        )

        gr.Markdown(
            """
            ### More details

            The model will generate two versions of the music based on different generation parameter presets.
            Two presets are selected based on their ELO ratings, and you can listen to both tracks and select which one you prefer.

            The model is fixed to `musicgen-medium`, and melody inputs are not used in this version.

            [Rest of the markdown content remains unchanged]
            """
        )

    interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    ui_full(launch_kwargs)
