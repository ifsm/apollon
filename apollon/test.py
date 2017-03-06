#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Michael Blaß'


from audio import AudioData
import segment
import tools

from extract import available_features
import extract

import model

print('Loading data ... ', end='')
try:
    x = AudioData('../audio/beat.wav')
    print('OK')
except:
    print('\nCould not load data. Exit.')
    raise

print('\nTesting segmentation methods')
try:
    print('by_samples ... ', end='')
    y = segment.by_samples(x, 1000)
    print('OK')

    print('by_samples_with_hop ... ', end='')
    y = segment.by_samples_with_hop(x, 880, 440)
    print('OK')

    print('by_ms ... ', end='')
    y = segment.by_ms(x, 1000)
    print('OK')

    print('by_ms_with_hop ... ', end='')
    y = segment.by_ms_with_hop(x, 1000, 500)
    print('OK')
except:
    print('\nError while testing segmentation methods.')
    raise

print('\nTesting feature extraction')
try:
    for name, featfunc in available_features.items():
        print(name, ' ... ', end='')
        ext = featfunc(y)
        print('OK')
except:
    print('\nError while testing feature extraction')
    raise

print('\nTesting scaling ... ', end='')
try:
    ext = tools.scale(ext, 100)
    print('OK')
except:
    print('\nError during scaling test. Exit.')
    raise

print('\nTesting HMM methods')
try:
    print('nstate model ... ', end='')
    model.nstates(ext, [2, 3])
    print('OK')
except:
    print('\nError while testing HHM methods.')
    raise

print('\nTest ended successfully!')
