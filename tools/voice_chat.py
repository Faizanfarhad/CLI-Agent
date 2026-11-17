# if failed then try to install the "alsa-utils" package and run again 

from pocket_tts import TTSModel, export_model_state
import scipy.io.wavfile
import subprocess
from typing import Optional 
from pathlib import Path

from config import OWN_VOICE_PATH

print(OWN_VOICE_PATH)
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
    
    model = TTSModel.load_model()
    
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
    if Path(own_voice).exists or own_voice in available_voices:
        model_state = model.get_state_for_audio_prompt(own_voice)
    else:
        model_state = model.get_state_for_audio_prompt("eve")
        
    export_model_state(model_state, "testing/some_voice.safetensors")
        
    # Later, load it quickly, this is quite fast as it's just reading the kvcache
    # from disk and doesn't do any others computations.
    model_state_copy = model.get_state_for_audio_prompt("testing/some_voice.safetensors")

    audio = model.generate_audio(model_state_copy, text)
    scipy.io.wavfile.write("testing/output.wav", model.sample_rate, audio.numpy())

    subprocess.run(["aplay", "testing/output.wav"])
    
    
# speak_model("as a indian my own voice copy can't able to talk in english lol","testing/myvoice.wav")