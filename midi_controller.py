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
tracks = (0, )
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

vocab_file = "Piano_vocab.npy"
vocab = np.load(os.path.join("vocab_save", vocab_file), allow_pickle=True)
notes_vocab = vocab[0]
durations_vocab = vocab[1]
offsets_vocab = vocab[2]
velocities_vocab = vocab[3]

print("\nVocab loaded\n", flush=True)

model_file = "Piano.h5"
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
        self.track.events.append(event)
        if status & 0xF0 == NOTE_ON:
            event.type = "NOTE_ON"
        else:   
            event.type = "NOTE_OFF"
            stream = midi.translate.midiTrackToStream(self.track).flat.notesAndRests
            if len(stream) >= seq_len:
                self.process_stream(stream[-seq_len:])
        print("[%s] @%0.6f %r" % (self.port, self._wallclock, message), flush=True)

    def process_stream(self, stream):
        notes = [[] for track in tracks]
        durations = [[] for track in tracks]
        offsets = [[] for track in tracks]
        velocities = [[] for track in tracks]
        for track, _ in enumerate(tracks):
            last_offset = 0
            for elem in stream:
                if isinstance(elem, note.Note):
                    notes[track].append([str(elem.pitch)])
                    durations[track].append(elem.quarterLength)
                    offsets[track].append(elem.offset - last_offset)
                    velocities[track].append(elem.volume.velocity)
                elif isinstance(elem, chord.Chord):
                    notes[track].append([str(n.nameWithOctave) for n in elem.pitches])
                    durations[track].append(elem.quarterLength)
                    offsets[track].append(elem.offset - last_offset)
                    velocities[track].append(elem.volume.velocity)
                elif isinstance(elem, note.Rest):
                    notes[track].append([elem.name])
                    durations[track].append(elem.quarterLength)
                    offsets[track].append(elem.offset - last_offset)
                    velocities[track].append(0)
                last_offset = elem.offset
        # Categorical
        cat_notes = [[] for track in tracks]
        cat_durations = [[] for track in tracks]
        cat_offsets = [[] for track in tracks]
        cat_velocities = [[] for track in tracks]
        for track, _ in enumerate(tracks):
            for elem in notes[track]:
                int_note = notes_vocab[track].index(",".join(elem))
                cat = np.zeros((len(notes_vocab[track])))
                cat[int_note] = 1
                cat_notes[track].append(cat)
            for elem in durations[track]:
                int_duration = 1#durations_vocab[track].index(elem)
                cat = np.zeros((len(durations_vocab[track])))
                cat[int_duration] = 1
                cat_durations[track].append(cat)
            for elem in offsets[track]:
                int_offset = 1#offsets_vocab[track].index(elem)
                cat = np.zeros((len(offsets_vocab[track])))
                cat[int_offset] = 1
                cat_offsets[track].append(cat)
            for elem in velocities[track]:
                int_velocity = 100#velocities_vocab[track].index(elem)
                cat = np.zeros((len(velocities_vocab[track])))
                cat[int_velocity] = 1
                cat_velocities[track].append(cat)
        # merge
        x = [cat_notes, cat_durations, cat_offsets, cat_velocities]
        # Prediction
        pred = model.predict([np.array(x[i]) for i in range(len(x))])
        # process predicted note
        for track, _ in enumerate(tracks):
            str_note = notes_vocab[track][np.argmax(pred[track][0][0])]
            _duration = durations_vocab[track][np.argmax(pred[track][0][1])]
            _offset = offsets_vocab[track][np.argmax(pred[track][0][2])]
            _velocity = velocities_vocab[track][np.argmax(pred[track][0][3])]
            if len(str_note.split(",")) > 1:
                _chord = chord.Chord(str_note.split(","))
                _chord.quarterLength = _duration
                _chord.offset = _offset
                _chord.volume.velocity = _velocity
                eventList = midi.translate.chordToMidiEvents(_chord)
                for event in eventList:
                    print(event, flush=True)
                    message = [0, 0, 0]
                    if event.type == "NOTE_ON": 
                        message[0] = NOTE_ON
                    elif event.type == "NOTE_OFF":
                        message[0] = NOTE_OFF
                    message[1] = event.pitch
                    message[2] = event.velocity
                    midiout.send_message(message)
            else:
                if str_note != "rest":
                    _note = note.Note(str_note)
                    _note.quarterLength = _duration
                    _note.offset = _offset
                    _note.volume.velocity = _velocity
                    eventList = midi.translate.noteToMidiEvents(_note)
                    for event in eventList:
                        print(event, flush=True)
                    message = [0, 0, 0]
                    if event.type == "NOTE_ON": 
                        message[0] = NOTE_ON
                    elif event.type == "NOTE_OFF":
                        message[0] = NOTE_OFF
                    message[1] = event.pitch
                    message[2] = event.velocity
                    midiout.send_message(message)
                else:
                    _rest = note.Rest()
                    _rest.quarterLength = _duration
                    _rest.offset = _offset
                    eventList = midi.translate.noteToMidiEvents(_rest)
                    for event in eventList:
                        print(event, flush=True)
                    message = [0, 0, 0]
                    if event.type == "NOTE_ON": 
                        message[0] = NOTE_ON
                    elif event.type == "NOTE_OFF":
                        message[0] = NOTE_OFF
                    message[1] = event.pitch
                    message[2] = event.velocity
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
