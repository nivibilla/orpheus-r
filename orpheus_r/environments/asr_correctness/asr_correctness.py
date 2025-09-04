import verifiers as vf
from datasets import load_dataset
import requests
import base64
import numpy as np
from openai import OpenAI
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
import editdistance


def load_environment(
    stt_client_url,
    audio_decoder_url,
    api_token,
    **kwargs
) -> vf.Environment:
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("srinivasbilla/orpheus-r-10k", split='train')

    def correct_answer(completion, answer):
        headers = {
            "Authorization": f"Bearer {api_token}"
        }
        data = {
            "generated_text": completion
        }
        decoded_audio_response = requests.post(audio_decoder_url, headers=headers, json=data)
        base64_string = decoded_audio_response.json()['audio_array_base64']
        audio_bytes = base64.b64decode(base64_string)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        stt_client = OpenAI(base_url=stt_client_url, api_key=api_token)

        raw_audio = RawAudio.from_audio(Audio(
                      audio_array=audio_array,
                      sampling_rate=24000,
                      format="wav",
                  ))
        
        transcribed_text = stt_client.audio.transcriptions.create(
            **TranscriptionRequest(
                model="voxtral-speech-detection", audio=raw_audio, language="en", temperature=0.0
            ).to_openai(exclude=("top_p", "seed"))
        ).text

        difference = editdistance.eval(transcribed_text, answer)
        return round((difference / len(answer))**2, 2)
    
    rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])

    return vf.SingleTurnEnv(
        dataset=ds,
        rubric=rubric,
        **kwargs
    )


vf.load_environment()
