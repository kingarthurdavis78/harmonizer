import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft


def sin_wave(frequency, amplitude, x):
    return amplitude * np.sin(2 * np.pi * frequency * x)


def saw(frequency, amplitude, x):
    return amplitude * (2 * np.mod(frequency * x, 1) - 1)

def square(frequency, amplitude, x):
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * x))

def triangle(frequency, amplitude, x):
    return amplitude * np.abs(saw(frequency, 1, x)) - amplitude / 2

def piano(frequency, amplitude, x):
    return amplitude * np.sin(2 * np.pi * frequency * x) + 0.5 * amplitude * np.sin(
        4 * np.pi * frequency * x) + 0.25 * amplitude * np.sin(6 * np.pi * frequency * x) + 0.125 * amplitude * np.sin(
        8 * np.pi * frequency * x)

def organ(frequency, amplitude, x):
    wave = np.zeros(len(x))
    for i in range(1, 10):
        wave += np.sin(2 * np.pi * frequency * x * i) / i
    return amplitude * wave

SAMPLE_RATE = 44100

# choose a wave function
WAVE = sin_wave

# plot one period of the wave
frequency = 440
amplitude = 1
duration = 1 / frequency
x = np.arange(0, duration, 1 / SAMPLE_RATE)
wave = WAVE(frequency, amplitude, x)

plt.plot(x, wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'One Period of a {WAVE.__name__.upper()} Wave')
plt.show()

# plot FFT of the sine wave to the right
FFT = fft(wave)
freqs = np.linspace(0, SAMPLE_RATE, len(FFT))
# plot bar chart
plt.bar(freqs[:len(FFT) // 2], np.abs(FFT)[:len(FFT) // 2] * 1 / len(FFT), width=5)

# cap the x axis at highest note on a piano
plt.xlim(0, 4186)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT')

plt.tight_layout()
plt.show()


