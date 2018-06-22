#!/data/hisparc/pgunnink/miniconda3/envs/kascade/bin/python


import tables
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import progressbar

"""
Find the noise levels for the Kascade data using the first 100 datapoints in every 
trace
"""

VERBOSE = False
path_to_kascade = '/data/hisparc/pgunnink/MachineLearning/Kascade/kascade.h5'
with tables.open_file(path_to_kascade,'r') as data:
    events = data.root.reconstructions_offsets
    entries = len(events)
    trace_noise = np.array((0,))
    if VERBOSE:
        iterator = progressbar.progressbar(events.iterrows(start=0, stop=entries),
                                       max_value=entries, redirect_stdout=True)
    else:
        iterator = events.iterrows(start=0, stop=entries)
    for row in iterator:
        traces = row['traces']
        padded = row['traces_padded']
        q = 0
        for trace in traces.T:
            if padded[q]:
                trace = trace[1200:]
            noise = trace[:100]- np.mean(trace[:100])
            trace_noise = np.append(trace_noise,noise)
            q += 1
    print(trace_noise.shape)
    plt.figure()
    h,bins,patches = plt.hist(trace_noise, bins=np.linspace(-2,2,30))
    plt.xlabel('Deviation from baseline')
    plt.ylabel('Occurence')
    plt.savefig('./plots/noisehistogram.png')
    plt.close()
    print('Mean: %s Standard deviation: %s' % (np.mean(trace_noise),np.std(trace_noise)))