"""Show how to receive MIDI input by setting a callback function."""

import logging
import sys
import time
from collections import deque

from rtmidi import MidiOut
from rtmidi.midiutil import open_midiinput
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF

seq_len = 32
bpm = 120/60
time_sign = [4, 4]

midiout = MidiOut()
available_ports = midiout.get_ports()
if available_ports:
    midiout.open_port(0)
else:
    midiout.open_virtual_port("My virtual output")

log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)


class MidiInputHandler(object):
    def __init__(self, port):
        self.port = port
        self.dq = deque(maxlen=seq_len)
        self._wallclock = 0
        self.curr_events = {}

    def __call__(self, event, data=None):
        message, deltatime = event
        # playback
        midiout.send_message(message)
        # process
        self.process_msg(message, deltatime)
        
    def process_msg(self, message, deltatime):
        self._wallclock += deltatime
        if message[0] & 0xF0 == NOTE_ON:
            status, note, velocity = message
            parsed_msg = [note, 0, deltatime * bpm, velocity]
            self.curr_events[note] = parsed_msg
        if message[0] & 0xF0 == NOTE_OFF:
            status, note, velocity = message
            self.curr_events[note][1] = deltatime * bpm
            self.dq.append(self.curr_events[note])
            self.curr_events.pop(note)
        
        print("[%s] @%0.6f %r" % (self.port, self._wallclock, message), flush=True)
        print(self.dq, flush=True)

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
