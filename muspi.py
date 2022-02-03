from __future__ import division
import pyaudio
from struct import pack
from math import cos, pi, floor, sin
import numpy as np
from random import randint

BLOCKSIZE = 2048
RATE = 44100
maxAmp = 2 ** 31 - 1.0
gain = 30000000
FLANGER = False
r = 44444
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt32,
                channels=2,
                rate=RATE,
                input=False,
                output=True)


def clip32(x):
    """Return clipped version of 32-bit signal."""
    if x > 2 ** 31 - 1:
        x = 2 ** 31 - 1
    elif x < -(2 ** 31 - 1):
        x = -(2 ** 31 - 1)
    else:
        x = x
    return int(x)


def exp_distortion(x, a=10.0):
    """Execute exponential soft-clipping and return distorted signal."""
    out = [0.0 for i in range(len(x))]
    m = max(x)

    if m == 0:
        return out

    x = [val / m for val in x]

    for j in range(len(x)):
        if x[j] > 0:
            out[j] = m * (1.0 - np.exp(-x[j] * a))
        else:
            out[j] = m * (-1.0 + np.exp(x[j] * a))
    return out


def merge(x, y):
    """Return merged list of input lists."""
    output = []

    for i in range(len(x)):
        val = (x[i] + y[i]) /2
        output.append(val)
        output.append(val)
    return output


def play(x, y=()):
    """Write to Pyaudio stream."""
    if len(y) is 0:
        y = x

    output = merge(x, y)
    output_string = pack('i' * len(output), *output)
    stream.write(output_string)


class Note:
    """Creates Note object in terms of note frequency and octave number."""

    # List of notes available
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # List of the fundamental frequencies of the notes C4 to B4 in Hertz
    frequencies = [261.626, 277.183, 293.665, 311.127, 329.628, 349.228, 369.994, 391.995, 415.305,
                   440.0, 466.164, 493.883]

    def __init__(self, name="C", octave_number=4):
        self.name = name
        self.octave_number = octave_number
        self.position = Note.notes.index(name)
        self.frequency = Note.frequencies[self.position] * 2.0 ** (self.octave_number - 4.0)

    def __add__(self, increment):
        """Overwritten + operator used for applying octave shift by increment amount to Note object."""


        name = Note.notes[(self.position + increment) % 12]
        octave_shift = (self.position + increment) // 12
        new_octave = self.octave_number + octave_shift
        new_note = Note(name, new_octave)
        return new_note

    def get_info(self):
        """Print note octave number and frequency."""

        print(self.name + str(self.octave_number) + ": " + str(self.frequency) + "Hz")


class Chord:
    """Creates Chord object that takes head Note object and adds appropriate notes based on c_type parameter."""

    def __init__(self, head_note, c_type):
        self.head_note = head_note
        self.type = c_type
        self.chords = self.get_chord_dict()

        assert self.type in self.chords.keys(), 'Invalid chord: %s' % self.type
        
        if self.type is 'silence':
            self.chord_notes = []
        else:
            self.chord_notes = [self.head_note]
            chord = self.chords[c_type]
            for c in chord:
                self.chord_notes.append(self.head_note + c)
    
    def get_chord_dict(self):
        """Return dict of chords"""
        chords = {
            'maj': [4, 7],
            'maj6': [4, 7, 9],
            'maj6add9': [4, 7, 9, 14],
            'add9': [4, 7, 14],
            'add11': [4, 7, 17],
            'maj7': [4, 7, 11],
            'maj9': [4, 7, 11, 14],
            'maj11': [4, 7, 11, 17],
            'maj13': [4, 7, 11, 21],
            'majb5': [4, 6],
            'maj7s11': [4, 7, 11, 18],
            'sus4': [5, 7],
            'sus2': [2, 7],
            '9sus4': [5, 7, 11, 14],
            'min': [3, 7],
            'min6': [3, 7, 9],
            'min6add9': [3, 7, 9, 14],
            'minadd9': [3, 7, 14],
            'min7': [3, 7, 10],
            'min7b5': [3, 6, 10],
            'min9': [3, 7, 10, 14],
            'min11': [3, 7, 10, 17],
            'min13': [3, 7, 10, 21],
            'dim': [3, 6],
            'dim4': [4, 6],
            'dim7': [3, 6, 10], # same as min7b5?
            'dimmaj': [4, 6], # same as dim4?
            'aug': [4, 8],
            'aug4': [5, 8],
            'augmin': [3, 8],
            '5': [7, 12],
            '4': [5],
            '7': [4, 7, 10],
            'note': [],
            'silence': []
        }
        return chords

    # noinspection PyUnusedLocal
    def play_chord(self, note_type=0.25, effect_list=[], tempo=200):
        """Method that plays Chord/Note/silence and applies any effect(s) passed in.
        Takes note type (quarter, half, etc.) and list of effects to be applied to that Chord/Note."""

        num_notes = len(self.chord_notes)
        Ta = (note_type * 4.0) * (60.0 / tempo)  # duration of chord
        # creates radius of pole based on note type and also account for if staccato effect is applied
        if 'staccato' in effect_list:
            r = 0.01 ** (1.0 / ((Ta / 2) * RATE))
        else:
            r = 0.01 ** (1.0 / ((Ta * RATE) / (note_type / 2)))
        w = []
        a1 = []
        a2 = r ** 2
        N = RATE * Ta
        num_blocks = int(floor(N / BLOCKSIZE))
        num_frames = num_blocks * BLOCKSIZE

        # creates constants for applying half bend and full bend effects
        if 'bend_half' in effect_list:
            bend_const = []
            for bent_note in self.chord_notes:
                bend_const.append(((bent_note + 1).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta ** 2))
        elif 'bend_full' in effect_list:
            bend_const = []
            for bent_note in self.chord_notes:
                bend_const.append(((bent_note + 2).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta ** 2))

        if FLANGER or 'vibrato' in effect_list or 'flanger' in effect_list:
            # initialization for vibrato parameters
            f0 = 5
            # Create a buffer (delay line) for past values
            BUFFER_LEN = BLOCKSIZE  # Buffer length
            buffer = [0.0 for i in range(BUFFER_LEN)]  # Initialize to zero
            # Buffer (delay line) indices
            kr = 0  # read index
            kw = int(0.5 * BUFFER_LEN)  # write index (initialize to middle of buffer)
            flanger_gain = 1  # gain for the vibrato if the flanger effect is active

        if 'echo' in effect_list:
            # Set parameters of delay system
            Gdp = 1.0  # direct-path gain
            Gff = 0.8  # feed-forward gain
            delay_sec = 0.15
            delay_samples = int(floor(RATE * delay_sec))
            BUFFER_LEN_ECHO = delay_samples  # set the length of buffer
            echo_buffer = [0 for v in range(BUFFER_LEN_ECHO)]
            k = 0  # read index

        for note in self.chord_notes:
            w.append(2 * pi * (note.frequency / RATE))
        for omega in w:
            a1.append(-2.0 * r * cos(omega))

        if num_notes is 0:  # play silence
            output = [0.0 for q1 in range(BLOCKSIZE)]
            for n1 in range(0, num_blocks):
                output_string = pack('i' * BLOCKSIZE, *output)
                stream.write(output_string)
            return 0

        # gain to equalize volume regardless of the number of notes per chord.
        g = gain / (num_notes ** .5)

        # Initialize zipper buffer
        zipper = []
        zipper2 = []
        for q2 in range(num_notes):
            zipper.append([0.0, 0.0])
            zipper2.append([0.0, 0.0])

        # Loop through blocks
        for n in range(0, num_blocks):
            data = [0.0 for q in range(BLOCKSIZE)]

            for x in range(0, BLOCKSIZE):
                current_frame = float(n * BLOCKSIZE + x)
                final_output = 0.0

                # generate current value of impulse input signal
                if n is 0 and x is 0:
                    x0 = 1.0
                else:
                    x0 = 0.0

                for i in range(num_notes):

                    # applies bend effect by increasing original frequency to next note (half bend)
                    # or two notes up (full bend) using a quadratic incrementation
                    if ('bend_full' in effect_list) or ('bend_half' in effect_list):
                        if current_frame < num_frames / 2:
                            a1[i] = -2.0 * r * cos((w[i] + bend_const[i] *
                                                    (current_frame * Ta / (num_frames / 2)) ** 2))
                        else:
                            a1[i] = -2.0 * r * cos((w[i] + bend_const[i] * Ta ** 2))
                    # generate impulse response for every note in the chord b
                    output = x0 - a1[i] * zipper[i][1] - a2 * zipper[i][0]
                    output2 = output - a1[i] * zipper2[i][1] - a2 * zipper2[i][0]
                    # sum of signal values for all notes for the total chord signal
                    final_output += zipper2[i].pop(0)
                    zipper[i].append(output)
                    zipper2[i].append(output2)

                # Apply gain and clip for the final chord signal value
                data[x] = clip32(g * final_output)
            if 'distortion' in effect_list:
                if num_notes == 1:
                    data = exp_distortion(data, 9)
                else:
                    data = exp_distortion(data, 1.5)
                    for v in range(len(data)):
                        data[v] = clip32(2 * data[v])

            # applying vibrato/flanger effects
            if FLANGER or 'vibrato' in effect_list or 'flanger' in effect_list:
                output_vibrato = []

                for q in range(BLOCKSIZE):
                    c = float(n * BLOCKSIZE + q)
                    vib_delay = 0.5

                    # starting the vibrato a quarter of the way into the signal and increasing it exponentially
                    if 'flanger' not in effect_list and (not FLANGER):
                        vib_start = num_blocks * BLOCKSIZE / 4.0
                        if c < vib_start:
                            vib_delay = 0.0
                        else:
                            vib_delay = ((c - vib_start) / (num_frames - vib_start)) * 1.5

                    W = 0.1 * vib_delay

                    kr_prev = int(floor(kr))
                    kr_next = kr_prev + 1
                    frac = kr - kr_prev  # 0 <= frac < 1

                    if kr_next >= BUFFER_LEN:
                        kr_next -= BUFFER_LEN

                    output_value = (1 - frac) * buffer[kr_prev] + frac * buffer[kr_next]
                    output_vibrato.append(clip32(output_value))

                    buffer[kw] = data[q]
                    kr = kr + 1 + W * sin(2 * pi * f0 * c / RATE)

                    if kr >= BUFFER_LEN:
                        # End of buffer. Circle back to front.
                        kr = 0

                    kw += 1
                    if kw == BUFFER_LEN:
                        # End of buffer. Circle back to front.
                        kw = 0

                    # Add original signal to the vibrato signal for the flanger effect
                    if FLANGER or 'flanger' in effect_list:
                        output_vibrato[q] = clip32((data[q] + flanger_gain * output_vibrato[q]) / (flanger_gain + .75))

                data = output_vibrato

            if 'echo' in effect_list:
                for e in range(0, BLOCKSIZE):
                    # Implementing echo using feed-forward gain filter
                    echo_output = Gdp * data[e] + Gff * echo_buffer[k]

                    # Update buffer
                    echo_buffer[k] = data[e]
                    k += 1
                    if k >= BUFFER_LEN_ECHO:
                        k = 0
                    data[e] = echo_output / (Gdp + Gff)

            data = merge(data, data)
            output_string = pack('i' * 2 * BLOCKSIZE, *data)
            stream.write(output_string)

    def get_info(self):
        """Print notes in each chord."""

        print(self.chord_notes[0].name +  self.type + ': '),
        for note in self.chord_notes:
            if note != self.chord_notes[-1]:
                print(note.name),
            else:
                print(note.name)


def play_bar(bar_l, bar_r=[], tempo=200):
    """Play one bar of music for both audio channels.
       Applies all audio effects."""
    if not bar_r:
        bar_r = bar_l[:]
    Ta_l = []
    Ta_r = []
    num_notes_l = []
    num_notes_r = []
    r_l = []
    r_r = []
    w_l = []
    w_r = []
    a1_l = []
    a1_r = []
    a2_l = []
    a2_r = []
    N_l = []
    N_r = []
    g_l = []
    g_r = []
    echo_l = []
    echo_r = []
    num_blocks_l = []
    num_blocks_r = []
    num_frames_l = []
    num_frames_r = []
    bend_const_l = []
    bend_const_r = []
    vibrato_l = []
    vibrato_r = []
    zipper1_l = []
    zipper1_r = []
    zipper2_l = []
    zipper2_r = []

    for chord in bar_l:
        Ta_l.append((chord[1] * 4.0) * (60.0 / tempo))  # duration of chord
        num_notes_l.append(len(chord[0].chord_notes))
        # creates radius of pole based on note type and also account for if staccato effect is applied
        if 'staccato' in chord[2]:
            r_l.append(0.01 ** (1.0 / ((Ta_l[-1] / 2) * RATE)))
        else:
            r_l.append(0.01 ** (1.0 / ((2 * Ta_l[-1] * RATE) / chord[1])))
        w = []
        a1 = []
        zipper1 = []
        zipper2 = []
        for note in chord[0].chord_notes:
            zipper1.append([0.0, 0.0])
            zipper2.append([0.0, 0.0])
            w.append(2 * pi * (note.frequency / RATE))
        for omega in w:
            a1.append(-2.0 * r_l[-1] * cos(omega))
        w_l.append(w)
        a1_l.append(a1)
        zipper1_l.append(zipper1)
        zipper2_l.append(zipper2)
        a2_l.append(r_l[-1] ** 2)
        N_l.append(RATE * Ta_l[-1])
        num_blocks_l.append(int(floor(N_l[-1] / BLOCKSIZE)))
        num_frames_l.append(num_blocks_l[-1] * BLOCKSIZE)

        if num_notes_l[-1] != 0:
            g_l.append(gain / (num_notes_l[-1] ** .3))
        else:
            g_l.append(1)

        # creates constants for applying half bend and full bend effects
        bend_const = []
        if 'bend_half' in chord[2]:
            for bent_note in chord[0].chord_notes:
                bend_const.append(((bent_note + 1).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta_l[-1] ** 2))
        elif 'bend_full' in chord[2]:
            for bent_note in chord[0].chord_notes:
                bend_const.append(((bent_note + 2).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta_l[-1] ** 2))
        bend_const_l.append(bend_const)

        # initialization for vibrato parameters
        vibrato = []
        if 'vibrato' in chord[2] or 'flanger' in chord[2]:
            f0 = 5
            # Create a buffer (delay line) for past values
            VIBRATO_BUFFER_LEN = BLOCKSIZE  # Buffer length
            buffer = [0.0 for i in range(VIBRATO_BUFFER_LEN)]
            # Buffer (delay line) indices
            kr = 0  # read index
            kw = int(0.5 * VIBRATO_BUFFER_LEN)  # write index (initialize to middle of buffer)
            flanger_gain = 0.3  # gain for the vibrato if the flanger effect is active
            vibrato = [f0, buffer, kr, kw, flanger_gain, VIBRATO_BUFFER_LEN]
        vibrato_l.append(vibrato)

        echo = []
        if 'echo' in chord[2]:
            # Set parameters of delay system
            Gdp = 1.0  # direct-path gain
            Gff = 0.8  # feed-forward gain
            delay_sec = 0.25
            delay_samples = int(floor(RATE * delay_sec))
            ECHO_BUFFER_LEN = delay_samples  # set the length of buffer
            echo_buffer = [0 for v in range(ECHO_BUFFER_LEN)]
            k = 0  # read index
            echo = [Gdp, Gff, delay_sec, k, echo_buffer, ECHO_BUFFER_LEN]
        echo_l.append(echo)

    for chord in bar_r:
        Ta_r.append((chord[1] * 4.0) * (60.0 / tempo))  # duration of chord
        num_notes_r.append(len(chord[0].chord_notes))
        # creates radius of pole based on note type and also account for if staccato effect is applied
        if 'staccato' in chord[2]:
            r_r.append(0.01 ** (1.0 / ((Ta_r[-1] / 2) * RATE)))
        else:
            r_r.append(0.01 ** (1.0 / ((2 * Ta_r[-1] * RATE) / chord[1])))
        w = []
        a1 = []
        zipper1 = []
        zipper2 = []
        for note in chord[0].chord_notes:
            zipper1.append([0.0, 0.0])
            zipper2.append([0.0, 0.0])
            w.append(2 * pi * (note.frequency / RATE))
        for omega in w:
            a1.append(-2.0 * r_l[-1] * cos(omega))
        w_r.append(w)
        a1_r.append(a1)
        zipper1_r.append(zipper1)
        zipper2_r.append(zipper2)
        a2_r.append(r_r[-1] ** 2)
        N_r.append(RATE * Ta_r[-1])
        num_blocks_r.append(int(floor(N_r[-1] / BLOCKSIZE)))
        num_frames_r.append(num_blocks_r[-1] * BLOCKSIZE)
        if num_notes_r[-1] != 0:
            g_r.append(gain / (num_notes_r[-1] ** .3))
        # creates constants for applying half bend and full bend effects
        bend_const = []
        if 'bend_half' in chord[2]:
            for bent_note in chord[0].chord_notes:
                bend_const.append(((bent_note + 1).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta_l[-1] ** 2))
        elif 'bend_full' in chord[2]:
            for bent_note in chord[0].chord_notes:
                bend_const.append(((bent_note + 2).frequency - bent_note.frequency) * (2 * pi / RATE) / (Ta_l[-1] ** 2))
        bend_const_r.append(bend_const)

        # initialization for vibrato parameters
        vibrato = []
        if 'vibrato' in chord[2] or 'flanger' in chord[2]:
            f0 = 5
            # Create a buffer (delay line) for past values
            VIBRATO_BUFFER_LEN = BLOCKSIZE  # Buffer length
            buffer = [0.0 for i in range(VIBRATO_BUFFER_LEN)]
            # Buffer (delay line) indices
            kr = 0  # read index
            kw = int(0.5 * VIBRATO_BUFFER_LEN)  # write index (initialize to middle of buffer)
            flanger_gain = 0.3  # gain for the vibrato if the flanger effect is active_
            vibrato = [f0, buffer, kr, kw, flanger_gain, VIBRATO_BUFFER_LEN]
        vibrato_r.append(vibrato)

        echo = []
        if 'echo' in chord[2]:
            # Set parameters of delay system
            Gdp = 1.0  # direct-path gain
            Gff = 0.8  # feed-forward gain
            delay_sec = 0.05
            delay_samples = int(floor(RATE * delay_sec))
            ECHO_BUFFER_LEN = delay_samples  # set the length of buffer
            echo_buffer = [0 for v in range(ECHO_BUFFER_LEN)]
            k = 0  # read index
            echo = [Gdp, Gff, delay_sec, k, echo_buffer, ECHO_BUFFER_LEN]
        echo_r.append(echo)

    total_blocks = min(sum(num_blocks_l), sum(num_blocks_r))
    block_count_l = 0
    bar_elem_l = 0
    block_count_r = 0
    bar_elem_r = 0

    for n in range(total_blocks):
        data_l = [0.0 for q in range(BLOCKSIZE)]
        data_r = [0.0 for q3 in range(BLOCKSIZE)]

        for x_l in range(BLOCKSIZE):
            current_frame = float(block_count_l * BLOCKSIZE + x_l)
            final_output_l = 0.0

            # generate current value of impulse input signal
            if block_count_l is 0 and x_l is 0:
                x0 = 1.0
            else:
                x0 = 0.0
            for i_l in range(num_notes_l[bar_elem_l]):

                # applies bend effect by increasing original frequency to next note (half bend)
                # or two notes up (full bend) using a quadratic incrementation
                if ('bend_full' in bar_l[bar_elem_l][2]) or ('bend_half' in bar_l[bar_elem_l][2]):
                    if current_frame < num_frames_l[bar_elem_l] / 2:
                        a1_l[bar_elem_l][i_l] = -2.0 * r_l[bar_elem_l] * \
                                                cos((w_l[bar_elem_l][i_l] + bend_const_l[bar_elem_l][i_l] *
                                                     (current_frame * Ta_l[bar_elem_l] /
                                                      (num_frames_l[bar_elem_l] / 2)) ** 2))
                    else:
                        a1_l[bar_elem_l][i_l] = -2.0 * r_l[bar_elem_l] * cos((w_l[bar_elem_l][i_l] +
                                                                              bend_const_l[bar_elem_l][i_l] *
                                                                              Ta_l[bar_elem_l] ** 2))

                # applying two filters to output to give signal a rise and decay
                output = x0 - a1_l[bar_elem_l][i_l] * zipper1_l[bar_elem_l][i_l][1] - a2_l[bar_elem_l] \
                                                    * zipper1_l[bar_elem_l][i_l][0]
                output2 = output - a1_l[bar_elem_l][i_l] * zipper2_l[bar_elem_l][i_l][1] - a2_l[bar_elem_l] \
                                                         * zipper2_l[bar_elem_l][i_l][0]

                final_output_l += zipper2_l[bar_elem_l][i_l].pop(0)
                zipper1_l[bar_elem_l][i_l].append(output)
                zipper2_l[bar_elem_l][i_l].append(output2)

            data_l[x_l] = clip32(g_l[bar_elem_l] * final_output_l)

        if 'distortion' in bar_l[bar_elem_l][2]:
            if num_notes_l[bar_elem_l] == 1:
                data_l = exp_distortion(data_l, 10)
            else:
                data_l = exp_distortion(data_l, 1.5)
                for v_l in range(len(data_l)):
                    data_l[v_l] = clip32(2 * data_l[v_l])


        # applying vibrato/flanger effects
        if 'vibrato' in bar_l[bar_elem_l][2] or 'flanger' in bar_l[bar_elem_l][2]:
            output_vibrato = []

            for q1 in range(BLOCKSIZE):
                c_l = float(block_count_l * BLOCKSIZE + q1)
                vib_delay = 1

                # starting the vibrato a quarter of the way into the signal
                if 'flanger' not in bar_l[bar_elem_l][2]:
                    vib_start = num_frames_l[bar_elem_l] / 4.0
                    if c_l < vib_start:
                        vib_delay = 0.0
                    else:
                        vib_delay = ((c_l - vib_start) / (num_frames_l[bar_elem_l] - vib_start)) ** 1.5

                W = 0.1 * vib_delay

                kr_prev = int(floor(vibrato_l[bar_elem_l][2]))
                kr_next = kr_prev + 1
                frac = vibrato_l[bar_elem_l][2] - kr_prev

                if kr_next >= vibrato_l[bar_elem_l][5]:
                    kr_next -= int(vibrato_l[bar_elem_l][2])

                output_value = (1 - frac) * vibrato_l[bar_elem_l][1][kr_prev] + frac * vibrato_l[bar_elem_l][1][kr_next]
                output_vibrato.append(clip32(output_value))

                vibrato_l[bar_elem_l][1][vibrato_l[bar_elem_l][3]] = data_l[q1]
                vibrato_l[bar_elem_l][2] = vibrato_l[bar_elem_l][2] + 1 + W * sin(
                    2 * pi * vibrato_l[bar_elem_l][0] * c_l / RATE)

                if vibrato_l[bar_elem_l][2] >= vibrato_l[bar_elem_l][5]:
                    vibrato_l[bar_elem_l][2] = 0

                vibrato_l[bar_elem_l][3] += 1
                if vibrato_l[bar_elem_l][3] == vibrato_l[bar_elem_l][5]:
                    vibrato_l[bar_elem_l][3] = 0

                # Add original signal to the vibrato signal for the flanger effect
                if 'flanger' in bar_l[bar_elem_l][2]:
                    output_vibrato[q1] = clip32((data_l[q1] + vibrato_l[bar_elem_l][4] * output_vibrato[q1]))

            data_l = output_vibrato

        if 'echo' in bar_l[bar_elem_l][2]:
            for e in range(0, BLOCKSIZE):

                # Implementing echo using feed-forward gain filter
                echo_output = echo_l[bar_elem_l][0] * data_l[e] + echo_l[bar_elem_l][1] * echo_l[bar_elem_l][4][
                    echo_l[bar_elem_l][3]]

                # Update buffer
                echo_l[bar_elem_l][4][echo_l[bar_elem_l][3]] = data_l[e]
                echo_l[bar_elem_l][3] += 1
                if echo_l[bar_elem_l][3] >= echo_l[bar_elem_l][5]:
                    echo_l[bar_elem_l][3] = 0

                data_l[e] = echo_output / (echo_l[bar_elem_l][0] + echo_l[bar_elem_l][1])

        block_count_l += 1
        if block_count_l == num_blocks_l[bar_elem_l]:
            block_count_l = 0
            bar_elem_l += 1

        for x_r in range(BLOCKSIZE):
            current_frame = float(block_count_r * BLOCKSIZE + x_r)
            final_output_r = 0.0

            # generate current value of impulse input signal
            if block_count_r is 0 and x_r is 0:
                x0 = 1.0
            else:
                x0 = 0.0

            for i_r in range(num_notes_r[bar_elem_r]):

                # applies bend effect by increasing original frequency to next note (half bend)
                # or two notes up (full bend) using a quadratic incrementation
                if ('bend_full' in bar_r[bar_elem_r][2]) or ('bend_half' in bar_r[bar_elem_r][2]):
                    if current_frame < num_frames_r[bar_elem_r] / 2:
                        a1_r[bar_elem_r][i_r] = -2.0 * r_r[bar_elem_r] * cos(
                            (w_r[bar_elem_r][i_r] + bend_const_r[bar_elem_r][i_r] *
                             (current_frame * Ta_r[bar_elem_r] /
                              (num_frames_r[bar_elem_r] / 2)) ** 2))
                    else:
                        a1_r[bar_elem_r][i_r] = -2.0 * r_r[bar_elem_r] * cos(
                            (w_r[bar_elem_r][i_r] + bend_const_r[bar_elem_r][i_r] *
                             Ta_r[bar_elem_r] ** 2))

                # applying two filters to output to give signal a rise and decay
                output = x0 - a1_r[bar_elem_r][i_r] * zipper1_r[bar_elem_r][i_r][1] - a2_r[bar_elem_r] \
                                                    * zipper1_r[bar_elem_r][i_r][0]
                output2 = output - a1_r[bar_elem_r][i_r] * zipper2_r[bar_elem_r][i_r][1] - a2_r[bar_elem_r] \
                                                         * zipper2_r[bar_elem_r][i_r][0]

                final_output_r += zipper2_r[bar_elem_r][i_r].pop(0)
                zipper1_r[bar_elem_r][i_r].append(output)
                zipper2_r[bar_elem_r][i_r].append(output2)

            data_r[x_r] = clip32(g_r[bar_elem_r] * final_output_r)

        if 'distortion' in bar_r[bar_elem_r][2] and num_notes_r:
            if num_notes_r[bar_elem_r] == 1:
                data_r = exp_distortion(data_r, 10)
            else:
                data_r = exp_distortion(data_r, 1.5)
                for v_r in range(len(data_r)):
                    data_r[v_r] = clip32(2*data_r[v_r])

        # applying vibrato/flanger effects
        if 'vibrato' in bar_r[bar_elem_r][2] or 'flanger' in bar_r[bar_elem_r][2]:
            output_vibrato = []

            for q1 in range(BLOCKSIZE):
                c_r = float(block_count_r * BLOCKSIZE + q1)
                vib_delay = 1

                # starting the vibrato a quarter of the way into the signal
                if 'flanger' not in bar_r[bar_elem_r][2]:
                    vib_start = num_frames_r[bar_elem_r] / 4.0
                    if c_r < vib_start:
                        vib_delay = 0.0
                    else:
                        vib_delay = ((c_r - vib_start) / (num_frames_r[bar_elem_r] - vib_start)) ** 1.5

                W = 0.1 * vib_delay

                kr_prev = int(floor(vibrato_r[bar_elem_r][2]))
                kr_next = kr_prev + 1
                frac = vibrato_r[bar_elem_r][2] - kr_prev

                if kr_next >= vibrato_r[bar_elem_r][5]:
                    kr_next -= int(vibrato_r[bar_elem_r][2])

                output_value = (1 - frac) * vibrato_r[bar_elem_r][1][kr_prev] + frac * vibrato_r[bar_elem_r][1][kr_next]
                output_vibrato.append(clip32(output_value))

                vibrato_r[bar_elem_r][1][vibrato_r[bar_elem_r][3]] = data_r[q1]
                vibrato_r[bar_elem_r][2] = vibrato_r[bar_elem_r][2] + 1 + W * sin(
                    2 * pi * vibrato_r[bar_elem_r][0] * c_r / RATE)

                if vibrato_r[bar_elem_r][2] >= vibrato_r[bar_elem_r][5]:
                    vibrato_r[bar_elem_r][2] = 0

                vibrato_r[bar_elem_r][3] += 1
                if vibrato_r[bar_elem_r][3] == vibrato_r[bar_elem_r][5]:
                    vibrato_r[bar_elem_r][3] = 0

                # Add original signal to the vibrato signal for the flanger effect
                if 'flanger' in bar_r[bar_elem_r][2]:
                    output_vibrato[q1] = clip32((data_r[q1] + vibrato_r[bar_elem_r][4] * output_vibrato[q1]))

            data_r = output_vibrato

        if 'echo' in bar_r[bar_elem_r][2]:
            for e in range(0, BLOCKSIZE):

                # Implementing echo using feed-forward gain filter
                echo_output = echo_r[bar_elem_r][0] * data_r[e] + echo_r[bar_elem_r][1] * echo_r[bar_elem_r][4][
                    echo_r[bar_elem_r][3]]

                # Update buffer
                echo_r[bar_elem_r][4][echo_r[bar_elem_r][3]] = data_r[e]
                echo_r[bar_elem_r][3] += 1
                if echo_r[bar_elem_r][3] >= echo_r[bar_elem_r][5]:
                    echo_r[bar_elem_r][3] = 0

                data_r[e] = echo_output / (echo_r[bar_elem_r][0] + echo_r[bar_elem_r][1])

        block_count_r += 1
        if block_count_r == num_blocks_r[bar_elem_r]:
            block_count_r = 0
            bar_elem_r += 1

        play(data_l, data_r)


def gen_maj_scale(key):
    """Return major scale in key of Note passed in."""
    maj_scale = []

    for i in range(8):
        if i == 0:
            maj_scale.append(key)
        elif i is 3 or i is 7:
            maj_scale.append(maj_scale[-1] + 1)
        else:
            maj_scale.append(maj_scale[-1] + 2)
    return maj_scale


def gen_scale(key, scale_type='maj'):
    """Return a specific scale type by modifying major scale."""
    scale = gen_maj_scale(key)

    if scale_type == 'harm min':
        scale[2] += -1
        scale[5] += -1
    elif scale_type == 'mel min':
        scale[2] += -1
    elif scale_type == 'min':
        scale[2] += -1
        scale[5] += -1
        scale[6] += -1
    elif scale_type == 'double harm':
        scale[1] += -1
        scale[5] += -1
    elif scale_type == 'lyd aug':
        scale[3] += 1
        scale[4] += 1
    elif scale_type == 'enigmatic':
        scale[1] += -1
        scale[3] += 1
        scale[4] += 1
        scale[5] += 1
    elif scale_type == 'half dim':
        scale[2] += -1
        scale[4] += -1
        scale[5] += -1
        scale[6] += -1
    return scale


def gen_diat_harm(scale):
    """Return diatonic harmony."""
    diat_harm = []

    for i in range(7):
        maj_scale_comp = gen_maj_scale(scale[i])

        i2 = (i + 2) % 7
        i3 = (i + 4) % 7

        third = scale[i2].position - maj_scale_comp[2].position
        fifth = scale[i3].position - maj_scale_comp[4].position
        if third > 6:
            third -= 12
        elif third < -6:
            third += 12
        if fifth > 6:
            fifth -= 12
        elif fifth < -6:
            fifth += 12

        if third is -1 and fifth is -1:
            diat_harm.append(Chord(scale[i], 'dim'))
        elif third is -1 and fifth is 0:
            x = randint(0, 2)
            if x == 0:
                diat_harm.append(Chord(scale[i], 'min'))
            elif x == 1:
                diat_harm.append(Chord(scale[i], 'min6'))
            else:
                diat_harm.append(Chord(scale[i], 'min7'))
        elif third is -1 and fifth is 1:
            diat_harm.append(Chord(scale[i], 'augmin'))
        elif third is 0 and fifth is -1:
            diat_harm.append(Chord(scale[i], 'dimmaj'))
        elif third is 0 and fifth is 0:
            x = randint(0, 2)
            if x == 0:
                diat_harm.append(Chord(scale[i], 'maj'))
            elif x == 1:
                diat_harm.append(Chord(scale[i], 'maj7'))
            else:
                diat_harm.append(Chord(scale[i], 'maj6'))
        elif third is 0 and fifth is 1:
            diat_harm.append(Chord(scale[i], 'aug'))
        elif third is 1 and fifth is -1:
            diat_harm.append(Chord(scale[i], 'dim4'))
        elif third is 1 and fifth is 0:
            diat_harm.append(Chord(scale[i], 'sus4'))
        elif third is 1 and fifth is 1:
            diat_harm.append(Chord(scale[i], 'aug4'))
        else:
            diat_harm.append(Chord(scale[i], '5'))
    return diat_harm


def play_scale(scale):
    """Play all notes in one scale."""
    for note in scale:
        c = Chord(note, 'note')
        c.play_chord(.25, ['distortion'])
    for note in reversed(scale):
        c = Chord(note, 'note')
        c.play_chord(.25, ['distortion'])


def gen_bar_struct(bar_duration, harmony=False):
    '''Returns a randomly generated list of note lengths. The randomization process is different
    depending on whether the rhythm is for a harmony or melody'''
    bar_struct = []
    if harmony:
        repeat = randint(0, 1) == 0
        if repeat:
            x = 2 ** (-1 * randint(0, 3))
            while x > bar_duration:
                x = 2 ** (-1 * randint(1, 3))
            while bar_duration > 0:
                if x < bar_duration:
                    bar_struct.append(x)
                    bar_duration -= x
                else:
                    bar_struct.append(bar_duration)
                    bar_duration = 0
            return bar_struct
        while bar_duration > 0:
            x = 2 ** (-1 * randint(0, 3))
            while x > bar_duration:
                x = 2 ** (-1 * randint(0, 3))
            if x == 0.125 and bar_duration >= 0.25:
                bar_struct.append(x)
                bar_struct.append(x)
                bar_duration -= 2*x
            else:
                bar_struct.append(x)
                bar_duration -= x
        return bar_struct

    if randint(0, 3) > 0:
        triplet = True
    else:
        triplet = False

    while bar_duration > 0:

        x = 2 ** (-1 * randint(1, 3))
        while x > bar_duration:
            x = 2 ** (-1 * randint(1, 3))
        if triplet and randint(0, 1) == 0 and x == 0.25:
            for i in range(3):
                bar_struct.append(0.25/3)
            bar_duration -= x
        elif x == 0.125 and bar_duration >= 0.25:
            bar_struct.append(x)
            bar_struct.append(x)
            bar_duration -= 0.25
        else:
            bar_struct.append(x)
            bar_duration -= x

    return bar_struct


def gen_bar(bar_struct_l, bar_struct_r, diat_harm, scale, index, chord_prog, c_index):
    '''Returns a list of three things.
    The first two are lists (one for the harmony and one for the melody) of lists;
    each inner list contains the parameters needed to play one note/chord in the bar.
    The parameters are: The chord object to be played, the duration of the chord/note, and the effects added to chord/note.
    These parameters are determined by a given: rhythm structure, diatonic harmony, scale, and chord progression.
    It also takes into account the last chord and note played in the bar before it.
    The third thing is information about the last chord played in the bar for the next bar.
    '''
    bar_l = []
    bar_r = []
    Staccato = False
    if randint(0, 3) == 0:
        Staccato = True
    x = randint(1, 3)
    for i in range(len(bar_struct_l)):
        effect_list_l = ['distortion']
        if x == 0:
            effect_list_l.append('echo')
        if bar_struct_l[i] == .25 and Staccato:
            effect_list_l.append('staccato')
        if i == (len(bar_struct_l) - 1) and bar_struct_l[i] > .25:
            effect_list_l.append('vibrato')
            if randint(0, 2) == 0:
                effect_list_l.append('bend_full')
            elif randint(0, 2) == 1:
                effect_list_l.append('bend_half')

        bar_l.append([scale[index], bar_struct_l[i], effect_list_l])
        jump = np.random.poisson(2)

        if randint(0, 1) == 0:
            jump *= -1

        index += jump
        index %= 7

    effect_list_r = ['distortion']
    if x == 0:
        effect_list_r.append('echo')
    for i2 in range(len(bar_struct_r)):
        chord = chord_prog[c_index]
        bar_r.append([diat_harm[chord], bar_struct_r[i2], effect_list_r])

    c_index += 1
    c_index %= len(chord_prog)

    return [bar_l, bar_r, c_index]


def play_rand_piece(time_sig=(4, 4), tempo=200, key = '', scale_type = '', chord_prog = [], num_bars = 30):
    '''Generates a random piece of music based on these parameters:
    time signature (given as a list of two values, default to 4/4),
    tempo in bpm (taken as an integer, default to 200bpm),
    key (taken as a string of the key name in uppercase, default to a random key),
    scale type (taken as a string of the scale type name in lowercase, default to a random scale type),
    chord progression (given as a list of integers corresponding to the roman numeral of the chord minus one, default to a random progresion), and
    a number of bars to play for (taken as an integer and defaults to 30).
    '''

    # initialize all tones
    nc = Note('C', 4)
    nc_sh = Note('C#', 4)
    nd = Note('D', 4)
    nd_sh = Note('D#', 4)
    ne = Note('E', 4)
    nf = Note('F', 4)
    nf_sh = Note('F#', 4)
    ng = Note('G', 4)
    ng_sh = Note('G#', 4)
    na = Note('A', 4)
    na_sh = Note('A#', 4)
    nb = Note('B', 4)

    scale_types = ['maj', 'min', 'harm min', 'mel min', 'half dim', 'double harm', 'enigmatic', 'lyd aug']
    # Choose a scale type
    if not scale_type:
        scale2play = randint(0, len(scale_types) - 1)  # Choose a random scale type if not given one
        scale_type = scale_types[scale2play]
    elif scale_type not in scale_types:
        print('Error: invalid scale type. The valid scale types are: maj, min, harm min, mel min, half dim, double harm, enigmatic, lyd aug')
        return 1

    tones = [nc, nc_sh, nd, nd_sh, ne, nf, nf_sh, ng, ng_sh, na, na_sh, nb]
    # Choose key
    if not key:
        key2play = randint(0, len(tones) - 1)
    else:
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if key in keys:
            key2play = keys.index(key)
        else:
            print('Error: invalid key. The valid keys are: C, C#, D, D#, E, F, F#, G, G#, A, A#, B')
            return 1

    if key2play < 7:
        key = tones[key2play]
    else:
        key = tones[key2play] + -12

    scale = gen_scale(key, scale_type)  # Generate scale

    # Generate diatonic harmony based on selected key and scale type
    diat_harm = gen_diat_harm(scale)

    # Play scale
    print('Playing in key: ' + scale[0].name + ' ' + scale_type)
    #play_scale(scale)

    chord_progs = [[0, 5, 4, 3], [0, 4, 5, 2, 3], [5, 4, 3, 4], [0, 5, 3, 4], [0, 3, 5, 4], [0, 4, 3, 4]]
    if not chord_prog:
        chord_prog2play = randint(0, len(chord_progs)-1)  # Choose chord progression
        chord_prog = chord_progs[chord_prog2play]

    # Display progression info
    print('Using the chord progression: '),
    chord_numerals_maj = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    chord_numerals_min = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
    for i in range(len(chord_prog)):
        if 'maj' in diat_harm[chord_prog[i]].type:
            print(chord_numerals_maj[chord_prog[i]]),
        else:
            print(chord_numerals_min[chord_prog[i]]),
    print()


    # Display diatonic harmony
    print('Diatonic harmony for ' + key.name + ' ' + scale_type + ':')
    for chord in diat_harm:
         chord.get_info()


    index = chord_prog[0]  # Choose starting note

    # Generate notes to play
    notes = []
    for note in scale:
        notes.append(Chord(note, 'note'))

    # Calculate the duration of a bar
    bar_duration = time_sig[0] / time_sig[1]
    # Initialize indexes
    c_index = 0
    rhythm_index = 0
    melody_index = 0

    # Generate the rhythm of the harmony section
    bar_struct_harmony = []
    for il in range(2):
        bar_struct_harmony.append(gen_bar_struct(bar_duration, True))

    # Generate the rhythm of the melody section
    bar_struct_melody = []
    for im in range(2):
        bar_struct_melody.append(gen_bar_struct(bar_duration))

    # Play for a certain number of bars
    for i in range(num_bars):
        [bar_l, bar_r, c_index] = gen_bar(bar_struct_melody[melody_index], bar_struct_harmony[rhythm_index], diat_harm,
                                          notes, index, chord_prog, c_index)
        play_bar(bar_l, bar_r, tempo)
        rhythm_index += 1
        melody_index += 1
        rhythm_index %= len(bar_struct_harmony)
        melody_index %= len(bar_struct_melody)
