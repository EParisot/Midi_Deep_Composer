import logging
logging.basicConfig(level=logging.INFO)
import sys
import os

import numpy as np
from keras.models import load_model

from music21 import midi, converter, instrument, stream, note, chord

seq_len = 32
tracks = (0,)
instru = instrument.Piano()

# Load data from training
logging.info("Loading, please wait...")

vocab_file = "Piano_50.npy"
vocab = np.load(os.path.join("vocab_save", vocab_file), allow_pickle=True)
notes_vocab = vocab[0]
durations_vocab = vocab[1]
offsets_vocab = vocab[2]
velocities_vocab = vocab[3]

logging.info("Vocab loaded")

model_file = "Piano_20.h5"
model = load_model(os.path.join("weight_save", model_file))
model._make_predict_function()

logging.info("Model %s loaded" % model_file)

if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
    logging.error("No args")
    exit(0)

midi_file = sys.argv[1]
midi_part = converter.parse(midi_file)
# Parse the midi file by the notes/chords it contains
notes = [[] for track in tracks]
durations = [[] for track in tracks]
offsets = [[] for track in tracks]
velocities = [[] for track in tracks]
for track, _ in enumerate(tracks):
    notes_to_parse = midi_part[tracks[track]].flat.notesAndRests
    last_offset = 0
    for elem in notes_to_parse:
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
        
logging.info("Song %s Loaded" % midi_file)

# turn notes to integers
cat_notes = [[] for track in tracks]
cat_durations = [[] for track in tracks]
cat_offsets = [[] for track in tracks]
cat_velocities = [[] for track in tracks]
for track, _ in enumerate(tracks):
    for elem in notes[track]:
        try:
            int_note = notes_vocab[track].index(",".join(elem))
        except ValueError:
            n = 1
            while ",".join(elem[:-n]) not in notes_vocab[track]:
                n += 1
            int_note = notes_vocab[track].index(",".join(elem[:-n]))
        cat = np.zeros((len(notes_vocab[track])))
        cat[int_note] = 1
        cat_notes[track].append(cat)
    for elem in durations[track]:
        try:
            int_duration = durations_vocab[track].index(elem)
        except ValueError:
            int_duration = (np.abs(np.asarray(durations_vocab[track]) - elem)).argmin()
        cat = np.zeros((len(durations_vocab[track])))
        cat[int_duration] = 1
        cat_durations[track].append(cat)
    for elem in offsets[track]:
        try:
            int_offset = offsets_vocab[track].index(elem)
        except ValueError:
            int_offset = (np.abs(np.asarray(offsets_vocab[track]) - elem)).argmin()
        cat = np.zeros((len(offsets_vocab[track])))
        cat[int_offset] = 1
        cat_offsets[track].append(cat)
    for elem in velocities[track]:
        try:
            int_velocity = velocities_vocab[track].index(elem)
        except ValueError:
            int_velocity = (np.abs(np.asarray(velocities_vocab[track]) - elem)).argmin()
        cat = np.zeros((len(velocities_vocab[track])))
        cat[int_velocity] = 1
        cat_velocities[track].append(cat)
# merge
x = [cat_notes, cat_durations, cat_offsets, cat_velocities]

# build seed stream
x = [x[i][0] for i in range(len(x))]
x_stream = [stream.Stream() for track in tracks]
for track, _ in enumerate(tracks):
    for i in range(len(x[0])):
        str_note = notes_vocab[track][np.argmax(x[4*track][i])]
        _duration = durations_vocab[track][np.argmax(x[4*track+1][i])]
        _offset = offsets_vocab[track][np.argmax(x[4*track+2][i])]
        _velocity = velocities_vocab[track][np.argmax(x[4*track+3][i])]
        if len(str_note.split(",")) > 1:
            _chord = chord.Chord(str_note.split(","))
            _chord.quarterLength = _duration
            _chord.offset = _offset
            _chord.volume.velocity = _velocity
            x_stream[track].append(_chord)
        else:
            if str_note != "rest":
                _note = note.Note(str_note)
                _note.quarterLength = _duration
                _note.offset = _offset
                _note.volume.velocity = _velocity
                x_stream[track].append(_note)
            else:
                _rest = note.Rest()
                _rest.quarterLength = _duration
                _rest.offset = _offset
                x_stream[track].append(_rest)
    x_stream[track].insert(0, instru)

logging.info("Predicting, please wait...")

# make seq_len predictions from seed
preds = [[] for track in tracks]
for _ in range(4 * seq_len):
    pred = model.predict([np.array([x[i]]) for i in range(len(x))])
    _note = [pred[i] for i in range(0, len(pred), 4)]
    _duration = [pred[i] for i in range(1, len(pred), 4)]
    _offset = [pred[i] for i in range(2, len(pred), 4)]
    _velocity = [pred[i] for i in range(3, len(pred), 4)]
    for track, _ in enumerate(tracks):
        cat_note = np.zeros((len(notes_vocab[track])))
        _note[track] = np.argmax(_note[track])
        cat_note[_note[track]] = 1
        cat_duration = np.zeros((len(durations_vocab[track])))
        _duration[track] = np.argmax(_duration[track])
        cat_duration[_duration[track]] = 1
        cat_offset = np.zeros((len(offsets_vocab[track])))
        _offset[track] = np.argmax(_offset[track])
        cat_offset[_offset[track]] = 1
        cat_velocity = np.zeros((len(velocities_vocab[track])))
        _velocity[track] = np.argmax(_velocity[track])
        cat_velocity[_velocity[track]] = 1
        x[4*track] = x[4*track][1:]
        x[4*track] = list(x[4*track]) + [cat_note]
        x[4*track+1] = x[4*track+1][1:]
        x[4*track+1] = list(x[4*track+1]) + [cat_duration]
        x[4*track+2] = x[4*track+2][1:]
        x[4*track+2] = list(x[4*track+2]) + [cat_offset]
        x[4*track+3] = x[4*track+3][1:]
        x[4*track+3] = list(x[4*track+3]) + [cat_velocity]
        preds[track].append((cat_note, cat_duration, cat_offset, cat_velocity))

# Build predicted stream
y_stream = [stream.Stream() for track in tracks]
for track, _ in enumerate(tracks):
    for i in range(len(preds[track])):
        str_note = notes_vocab[track][np.argmax(preds[track][i][0])]
        _duration = durations_vocab[track][np.argmax(preds[track][i][1])]
        _offset = offsets_vocab[track][np.argmax(preds[track][i][2])]
        _velocity = velocities_vocab[track][np.argmax(preds[track][i][3])]
        if len(str_note.split(",")) > 1:
            _chord = chord.Chord(str_note.split(","))
            _chord.quarterLength = _duration
            _chord.offset = _offset
            _chord.volume.velocity = _velocity
            y_stream[track].append(_chord)
        else:
            if str_note != "rest":
                _note = note.Note(str_note)
                _note.quarterLength = _duration
                _note.offset = _offset
                _note.volume.velocity = _velocity
                y_stream[track].append(_note)
            else:
                _rest = note.Rest()
                _rest.quarterLength = _duration
                _rest.offset = _offset
                y_stream[track].append(_rest)
    y_stream[track].insert(0, instru)

logging.info("Playing")
# play generated music
y_full_score = stream.Score()
for track, _ in enumerate(tracks):
    p = stream.Part()
    p.append(y_stream[track])
    y_full_score.insert(0, p)
y_full_score.write("midi", midi_file.split(".mid")[0] + "_result.mid")
sp = midi.realtime.StreamPlayer(y_full_score)
sp.play()
