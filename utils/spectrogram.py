import matplotlib.figure
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

def audio_to_spectrogram(filename: str) -> matplotlib.figure.Figure:
    y, sr = librosa.load(filename)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    ret = plt.figure(figsize=(12,6))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    return ret

