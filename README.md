# Apollon

Apollon is a tool for music modelling. It comprises
* low-level audio feature extraction
* Hidden-Markov Models
* Self-Organizing Map

## 1. Installation
Download apollon or clone this repository. Navigate the packages root directory
and install apollon using pip.
```
cd path/to/apollon
pip install .
```
Note that the period on the end of the last line is necessary.

## 2. Commandline tools
Apollon comes with several commandline utilities implemented as subcommands 
of the main app. You may invoke the using
```apollon [subcommand]```

### 2.1 Feature extraction
#### 2.1.1 Timbre track
```apollon features --timbre audio_file [-o output_file]```

#### 2.1.2 Rhythm track
```apollon features --rhythm audio_file [- output_file]```
