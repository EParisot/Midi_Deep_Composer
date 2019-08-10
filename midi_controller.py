"""Show how to receive MIDI input by setting a callback function."""

import logging
import sys
import os
import time
from collections import deque
import numpy as np
from keras.models import load_model

from music21 import midi, instrument, stream, note, chord
from rtmidi import MidiOut
from rtmidi.midiutil import open_midiinput
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF

seq_len = 32
bpm = 120/60
instru = instrument.Piano()

# MIDI Setup
midiout = MidiOut()
available_ports = midiout.get_ports()
if available_ports:
    midiout.open_port(0)
else:
    midiout.open_virtual_port("My virtual output")

log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)

# Load data from training
print("\nLoading, please wait...\n", flush=True)

vocab_file = "Piano_50.npy"
vocab = np.load(os.path.join("vocab_save", vocab_file), allow_pickle=True)
notes_vocab = vocab[0]
durations_vocab = vocab[1]
offsets_vocab = vocab[2]
velocities_vocab = vocab[3]

print("\nVocab loaded\n", flush=True)

model_file = "Piano_50.h5"
model = load_model(os.path.join("weight_save", model_file))
model._make_predict_function()
print("\nModel %s loaded\n" % model_file, flush=True)

class MidiInputHandler(object):
    def __init__(self, port):
        self.port = port
        self._wallclock = 0
        self.track = midi.MidiTrack(1)

    def __call__(self, event, data=None):
        message, deltatime = event
        # playback
        midiout.send_message(message)
        # process
        self.process_midi(message, deltatime)

    def process_midi(self, message, deltatime):
        self._wallclock += deltatime
        status, note, velocity = message
        
        dt = midi.DeltaTime(self.track)
        dt.time = deltatime * 1000
        self.track.events.append(dt)
        
        event = midi.MidiEvent(self.track)
        event.channel = 1
        event.pitch = note
        event.velocity = velocity
        if status & 0xF0 == NOTE_ON:
            event.type = "NOTE_ON"
            self.track.events.append(event)
        else:   
            event.type = "NOTE_OFF"
            self.track.events.append(event)
            stream = midi.translate.midiTrackToStream(self.track).flat.notesAndRests
            if len(stream) >= seq_len:
                print("Go Predict", flush=True)
                self.process_stream(stream[-seq_len:])
                self.track = midi.MidiTrack(1)
        #print("[%s] @%0.6f %r" % (self.port, self._wallclock, message), flush=True)

    def process_stream(self, stream):
        notes = []
        durations = []
        offsets = []
        velocities = []
        last_offset = 0
        for elem in stream:
            if isinstance(elem, note.Note):
                notes.append([str(elem.pitch)])
                durations.append(elem.quarterLength)
                offsets.append(elem.offset - last_offset)
                velocities.append(elem.volume.velocity)
            elif isinstance(elem, chord.Chord):
                notes.append([str(n.nameWithOctave) for n in elem.pitches])
                durations.append(elem.quarterLength)
                offsets.append(elem.offset - last_offset)
                velocities.append(elem.volume.velocity)
            elif isinstance(elem, note.Rest):
                notes.append([elem.name])
                durations.append(elem.quarterLength)
                offsets.append(elem.offset - last_offset)
                velocities.append(0)
            last_offset = elem.offset
        # Categorical
        cat_notes = []
        cat_durations = []
        cat_offsets = []
        cat_velocities = []

        for elem in notes:
            int_note = notes_vocab[0].index(",".join(elem))
            cat = np.zeros((len(notes_vocab[0])))
            cat[int_note] = 1
            cat_notes.append(cat)
        for elem in durations:
            int_duration = 1#durations_vocab.index(elem)
            cat = np.zeros((len(durations_vocab[0])))
            cat[int_duration] = 1
            cat_durations.append(cat)
        for elem in offsets:
            int_offset = 1#offsets_vocab.index(elem)
            cat = np.zeros((len(offsets_vocab[0])))
            cat[int_offset] = 1
            cat_offsets.append(cat)
        for elem in velocities:
            int_velocity = 50#velocities_vocab.index(elem)
            cat = np.zeros((len(velocities_vocab[0])))
            cat[int_velocity] = 1
            cat_velocities.append(cat)
        # merge
        x = [cat_notes, cat_durations, cat_offsets, cat_velocities]
        # make seq_len predictions from seed
        preds = []
        for _ in range(seq_len):
            pred = model.predict([np.array([x[i]]) for i in range(len(x))])
            _note = [pred[i] for i in range(0, len(pred), 4)]
            _duration = [pred[i] for i in range(1, len(pred), 4)]
            _offset = [pred[i] for i in range(2, len(pred), 4)]
            _velocity = [pred[i] for i in range(3, len(pred), 4)]
            cat_note = np.zeros((len(notes_vocab[0])))
            _note = np.argmax(_note)
            cat_note[_note] = 1
            cat_duration = np.zeros((len(durations_vocab[0])))
            _duration = np.argmax(_duration)
            cat_duration[_duration] = 1
            cat_offset = np.zeros((len(offsets_vocab[0])))
            _offset = np.argmax(_offset)
            cat_offset[_offset] = 1
            cat_velocity = np.zeros((len(velocities_vocab[0])))
            _velocity = np.argmax(_velocity)
            cat_velocity[_velocity] = 1
            x[0] = x[0][1:]
            x[0] = list(x[0]) + [cat_note]
            x[1] = x[1][1:]
            x[1] = list(x[1]) + [cat_duration]
            x[2] = x[2][1:]
            x[2] = list(x[2]) + [cat_offset]
            x[3] = x[3][1:]
            x[3] = list(x[3]) + [cat_velocity]
            preds.append((cat_note, cat_duration, cat_offset, cat_velocity))
        # process predicted note
        for pred in preds:
            str_note = notes_vocab[0][np.argmax(pred[0])]
            _duration = durations_vocab[0][np.argmax(pred[1])]
            _offset = offsets_vocab[0][np.argmax(pred[2])]
            _velocity = velocities_vocab[0][np.argmax(pred[3])]
            if len(str_note.split(",")) > 1:
                _chord = chord.Chord(str_note.split(","))
                _chord.quarterLength = _duration
                _chord.offset = _offset
                _chord.volume.velocity = _velocity
                eventList = midi.translate.chordToMidiEvents(_chord)
                for event in eventList:
                    message = [0, 0, 0]
                    if event.type == "NOTE_ON": 
                        message[0] = NOTE_ON
                    elif event.type == "NOTE_OFF":
                        message[0] = NOTE_OFF
                    message[1] = event.pitch
                    message[2] = event.velocity
                    if event.type == "DeltaTime":
                        message[0] = NOTE_ON
                        message[1] = 21
                        message[2] = 0
                        time.sleep(event.time/1000)
                    midiout.send_message(message)
            else:
                if str_note != "rest":
                    _note = note.Note(str_note)
                    _note.quarterLength = _duration
                    _note.offset = _offset
                    _note.volume.velocity = _velocity
                    eventList = midi.translate.noteToMidiEvents(_note)
                    for event in eventList:
                        message = [0, 0, 0]
                        if event.type == "NOTE_ON": 
                            message[0] = NOTE_ON
                        elif event.type == "NOTE_OFF":
                            message[0] = NOTE_OFF
                        message[1] = event.pitch
                        message[2] = event.velocity
                        if event.type == "DeltaTime":
                            message[0] = NOTE_ON
                            message[1] = 21
                            message[2] = 0
                            time.sleep(event.time/1000)
                        midiout.send_message(message)
                else:
                    _rest = note.Rest()
                    _rest.quarterLength = _duration
                    _rest.offset = _offset
                    eventList = midi.translate.noteToMidiEvents(_rest)
                    for event in eventList:
                        message = [0, 0, 0]
                        if event.type == "NOTE_ON": 
                            message[0] = NOTE_ON
                        elif event.type == "NOTE_OFF":
                            message[0] = NOTE_OFF
                        message[1] = event.pitch
                        message[2] = event.velocity
                        if event.type == "DeltaTime":
                            message[0] = NOTE_ON
                            message[1] = 21
                            message[2] = 0
                            time.sleep(event.time/1000)
                        midiout.send_message(message)

        
# Prompts user for MIDI input port, unless a valid port number or name
# is given as the first argument on the command line.
# API backend defaults to ALSA on Linux.
port = sys.argv[1] if len(sys.argv) > 1 else None

try:
    midiin, port_name = open_midiinput(port)
except (EOFError, KeyboardInterrupt):
    sys.exit()

print("Attaching MIDI input callback handler.")
midiin.set_callback(MidiInputHandler(port_name))

print("Entering main loop. Press Control-C to exit.")
try:
    # Just wait for keyboard interrupt,
    # everything else is handled via the input callback.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('')
finally:
    print("Exit.")
    midiin.close_port()
    del midiin
    del midiout
