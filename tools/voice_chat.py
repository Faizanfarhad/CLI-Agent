###################TTS part

# if TTS failed then try to install the "alsa-utils" package and run again 
from pocket_tts import TTSModel, export_model_state
import scipy.io.wavfile
import subprocess
from typing import Optional 
from pathlib import Path

from config import OWN_VOICE_PATH

tts_model = TTSModel.load_model()

def speak_model(text:str,own_voice:Optional[str | None] = OWN_VOICE_PATH):
    """_summary_

    Args:
        text (str): the text that you want the model to spell 
        own_voice (Optional[str  |  None]): str should be the path of your own voice or ['alba' (en) for english
'giovanni' (it) for italin, 
'lola' (es) for spanish,
'juergen' (de) ,
'rafael' (pt),
'estelle' (fr) for french,
'anna' (en) for english,
'azelma' (en) for english,
'bill_boerst' (en) for english,
'caro_davy' (en) for english,
'charles' (en) for english,
'cosette' (en) for english,
'eponine' (en) for english,
'eve' (en) for english,
'fantine' (en) for english,
'george' (en) for english,
'jane' (en) for english,
'jean' (en) for english,
'javert' (en) for english,
'marius' (en) for english,
'mary' (en) for english,
'michael' (en) for english,
'paul' (en) for english,
'peter_yearsley' (en) for english,
'stuart_bell' (en) for english,
'vera' (en) for english
]
    """
    
    global tts_model
    
    available_voices = ['bill_boerst','caro_davy','charles','cosette','eponine','eve'
        'fantine',
        'george' ,
        'jane' ,
        'jean' ,
        'javert' ,
        'marius' ,
        'mary',
        'michael',
        'paul' ,
        'peter_yearsley',
        'stuart_bell' ,
        'vera' 
        ]
    
    
    # Export a voice state for fast loading later
    if Path(own_voice).exists() or own_voice in available_voices:
        model_state = tts_model.get_state_for_audio_prompt(own_voice)
    else:
        model_state = tts_model.get_state_for_audio_prompt("eve")
        
    export_model_state(model_state, "testing/some_voice.safetensors")
        
    # Later, load it quickly, this is quite fast as it's just reading the kvcache
    # from disk and doesn't do any others computations.
    model_state_copy = tts_model.get_state_for_audio_prompt("testing/some_voice.safetensors")

    audio = tts_model.generate_audio(model_state_copy, text)
    scipy.io.wavfile.write("testing/output.wav", tts_model.sample_rate, audio.numpy())
    
    result = subprocess.run(
        ["aplay", "testing/output.wav"],
        capture_output=True,
        text=True,
        )
    if result.returncode != 0:
        return (result.stderr or "Audio playback failed", False)
    return True

#################mic listener

from collections import deque
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad
from transformers import AutoProcessor
from faster_whisper import WhisperModel

from config import (
    ASR_MODEL_DIR,
    SAMPLE_RATE,
    PRE_BUFFER_CHUNKS,
    CHUNK_SAMPLES,
    SPEECH_THRESHOLD,
    SILENCE_CHUNKS
    )

MODEL_DIR = ASR_MODEL_DIR
device = "cpu"
dtype = torch.float32

vad_model = load_silero_vad()
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
asr_model = WhisperModel("small", device=device, compute_type="float32")


def transcribe(audio_np):
    segments, _ = asr_model.transcribe(audio_np, beam_size=1)
    return "".join(seg.text for seg in segments).strip()

def listen_loop():
    pre_buffer = deque(maxlen=PRE_BUFFER_CHUNKS)
    speech_buffer = []
    speaking = False
    silence_counter = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES) as stream:
        print("Listening...")
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk_1d = chunk.squeeze().copy()
            speech_prob = vad_model(torch.from_numpy(chunk_1d), SAMPLE_RATE).item()
            if speech_prob >= SPEECH_THRESHOLD:
                if not speaking:
                    speaking = True
                    speech_buffer.extend(pre_buffer)  # prepend audio right before trigger
                speech_buffer.append(chunk_1d)
                silence_counter = 0
            elif speaking:
                speech_buffer.append(chunk_1d)
                silence_counter += 1
                if silence_counter >= SILENCE_CHUNKS:
                    # Utterance finished -> run ASR
                    utterance = np.concatenate(speech_buffer)
                    text = transcribe(utterance)
                    if text:
                        print(f"You: {text}")
                    # Reset state
                    speaking = False
                    silence_counter = 0
                    speech_buffer.clear()
                    pre_buffer.clear()
                    return text
            else:
                pre_buffer.append(chunk_1d)

def listen_to_mic()->str:
    """_summary_
    model listen the user voice until the threshold of silence   
    Returns:
        _type_: str
    """

    text = listen_loop()
    return text
