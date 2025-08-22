import numpy as np
from scipy.io.wavfile import write

# Parameters
duration = 2.0        # seconds
freq = 1000           # Hz (pitch of the beep)
samplerate = 44100    # samples per second

# Generate time array
t = np.linspace(0, duration, int(samplerate * duration), False)

# Generate a sine wave (alarm tone)
tone = 0.5 * np.sin(2 * np.pi * freq * t)

# Convert to 16-bit PCM
audio = np.int16(tone * 32767)

# Save as WAV file
write("alarm.wav", samplerate, audio)

print("Alarm sound saved as 'alarm.wav'")
