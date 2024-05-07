# Raptor: The Random Arpeggiator

Copyright © 2024 by Albert Gräf \<<aggraef@gmail.com>\>, distributed under the GPL (see COPYING). Please also check my GitHub page at https://agraef.github.io/.

**In memory of Clarence Barlow** (27 December 1945 – 29 June 2023).

This is version 7 of the Raptor patch, an experimental arpeggiator program based on the mathematical music theories of composer and computer music pioneer extraordinaire Clarence Barlow. This version is a backport of the [Ardour plugin](https://github.com/agraef/ardour-lua) included in [Ardour version 8](https://ardour.org/news/8.1.html) and later, which in turn was based on the original Lua version of Raptor (version 6). The present version is compatible with the Ardour plugin in terms of the underlying arpeggiator core (written in Lua), as well as the parameters and factory presets. While it is ultimately based on [Raptor 6](https://github.com/agraef/ardour-lua), the patch was completely rewritten to provide an improved and simplified interface, and also offers some important new features, such as latch mode, improved transport and looper functions, a MIDI learn facility, and much improved built-in support for a bunch of popular MIDI controllers.

Raptor runs in any modern flavor of Pd. E.g., here's how the patch looks like in [Purr Data](https://agraef.github.io/purr-data/):

<img src="doc/raptor7.png" alt="raptor7" />

Please check the [manual](doc/raptor7.md) included in the distribution for detailed usage instructions. In a nutshell, you open raptor7.pd in Pd, hook up your MIDI keyboard and synthesizer to Pd's first MIDI input and output, respectively, choose a preset, press the green playback toggle in the time subpatch, and start playing chords. Note that Raptor only generates MIDI data, so you need some sound-generating device to make music with it. But it works just fine with a free software synth such as Qsynth. Of course, Raptor can also be used with a DAW or outboard gear such as a hardware synth or a groovebox, to which it can be synced via MIDI clocks. This is all discussed in detail in the manual.
