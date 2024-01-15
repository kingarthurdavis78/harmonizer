import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.fftpack import fft, ifft
import pygame.midi as midi
from matplotlib import pyplot as plt

midi.init()
device = midi.get_default_input_id()
midi_in = midi.Input(device)

SAMPLE_RATE = 48000
frames = 1024

stretch = 4 * SAMPLE_RATE / frames

prev_freq = 0

active_notes = {}


def sine(t, freq):
    return np.float32(np.sin(2 * np.pi * freq * t))


def midi_to_frequency(note):
    return 440 * 2 ** ((note - 69) / 12)


prev_waves = {}


def sine_one_period(freq):
    if freq == 0:
        return np.zeros(frames)
    t = np.arange(0, 1 / freq, 1 / SAMPLE_RATE)
    return sine(t, freq)


def repeat_wave_until_at_least_length(wave, length):
    new_wave = wave[:]
    while len(new_wave) < length:
        new_wave = np.concatenate((new_wave, wave))
    return new_wave


def harmonic(note, magnitude, index, volume):
    freq = midi_to_frequency(note)
    max_amp = magnitude[index]
    result = np.zeros(frames)

    num = (note - 21) / 108

    harmonics_count = 1
    if len(active_notes) != 0:
        harmonics_count = int(4 // num)

    for i in range(1, harmonics_count + 1):

        if index * i >= len(magnitude):
            break

        if freq * i > SAMPLE_RATE / 2:
            break

        if note not in prev_waves:
            prev_waves[note] = {}

        if i - 1 not in prev_waves[note]:
            prev_waves[note][i - 1] = np.zeros(frames)

        prev = prev_waves[note][i - 1]

        # check if prev wave is all zeros or nan
        if np.all(prev == 0) or np.isnan(prev).all():
            one_period = sine_one_period(freq * i)
            prev_waves[note][i - 1] = repeat_wave_until_at_least_length(one_period, frames) * (
                        magnitude[index * i] / max_amp) * volume
            result += prev_waves[note][i - 1][:frames]
            prev_waves[note][i - 1] = np.roll(prev_waves[note][i - 1], -frames)
            continue

        zero_crossings = np.where(np.diff(np.signbit(prev)))[0]
        if prev[zero_crossings[0] - 1] < 0:
            # pick random zero crossing that is even
            random_num = np.random.randint(0, len(zero_crossings) // 2)
            first_zero = zero_crossings[random_num * 2]
        else:
            random_num = np.random.randint(0, len(zero_crossings) // 2)
            first_zero = zero_crossings[random_num * 2 + 1]

        one_period = sine_one_period(freq * i)
        new_wave = repeat_wave_until_at_least_length(one_period, frames) * (magnitude[index * i] / max_amp) * volume

        i_harmonic = np.concatenate((prev[:first_zero], new_wave[:frames - first_zero]))
        prev_waves[note][i - 1] = np.roll(new_wave, -(frames - first_zero))

        result += i_harmonic[:frames]

    return result


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    original = np.squeeze(indata)

    # find volume
    volume = np.max(np.abs(original))


    FFT = fft(original, int(frames * stretch))
    magnitude = np.abs(FFT[:len(FFT) // 2])

    index = np.argmax(magnitude)

    if midi_in.poll():
        midi_data = midi_in.read(1)[0]
        note_data, time = midi_data
        note = note_data[1]
        velocity = note_data[2]
        if velocity == 0:
            if note in active_notes:
                del active_notes[note]
        else:
            active_notes[note] = (velocity / 127) * (1 - note / 108)

    result = np.zeros(frames)
    for note in active_notes:
        if note < 29:
            continue
        note_volume = active_notes[note]
        result += volume * harmonic(note, magnitude, index, note_volume * volume)

    wave = result[:frames] * 0.5

    wave += original * 0.1

    outdata[:] = wave.reshape((len(wave), 1))


with sd.Stream(channels=1, device=(3, 1), callback=callback, samplerate=SAMPLE_RATE, dtype=np.float32,
               blocksize=frames):
    print(sd.query_devices())
    input("Press enter to quit:")
