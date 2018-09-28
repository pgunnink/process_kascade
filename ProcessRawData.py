#!/data/hisparc/pgunnink/miniconda3/envs/kascade/bin/python
#PBS -q short
#PBS -d /data/hisparc/pgunnink/Kascade/process_kascade

import tables
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import progressbar
from sapphire.analysis.find_mpv import FindMostProbableValueInSpectrum
from pathlib import Path

from ProcessDataML.DegRad import azimuth_zenith_to_cartestian


path_to_kascade = '/data/hisparc/pgunnink/MachineLearning/Kascade/kascade.h5'
path_to_new_file = '/data/hisparc/pgunnink/MachineLearning/Kascade/ML_file_testing.h5'


ADC_THRESHOLD = 20
ADC_TIME_PER_SAMPLE = 2.5
N_stations = 1
VERBOSE = True
N_TIME_BINS = 100
PLOT = False
overwrite = True
steps = 1

def get_time_in_ns(trace):
    return np.linspace(0,np.max(trace.shape)*ADC_TIME_PER_SAMPLE,np.max(trace.shape))

def find_closest(bins, value):
    for i in range(len(bins)-1):
        if value>=bins[i] and value<bins[i+1]:
            return i

class Events(tables.IsDescription):
    id = tables.Int32Col()
    traces = tables.Float32Col(shape=(N_stations * 4, 80))
    labels = tables.Float32Col(shape=3)
    timings = tables.Float32Col(shape=4)
    pulseheights = tables.Float32Col(shape=4)
    integrals = tables.Float32Col(shape=4)
    mips = tables.Float32Col(shape=4)
    rec_z = tables.Float32Col()
    rec_a = tables.Float32Col()
    zenith = tables.Float32Col()
    azimuth = tables.Float32Col()
    timestamp = tables.Time32Col()
    core_distance = tables.Float32Col()
    mpv = tables.Float32Col()
    energy = tables.Float32Col()

with tables.open_file(path_to_new_file, 'w') as f:
    group = f.create_group('/', 'kascade', 'Kascade data')
    table = f.create_table(group, 'events', Events, 'Events')
    new_row = table.row
    with tables.open_file(path_to_kascade,'r') as data:
        # events with offsets corrected are stored in /reconstructions_offsets
        events = data.root.reconstructions_offsets_complete
        entries = len(events)
        if VERBOSE:
            print("Total entries before filtering: %s" % entries)

        # first create an mpv fit per 8 hours, which is used later to normalize the
        # traces
        times = events.col('timestamp')
        min_time = times.min()
        max_time = times.max()+1
        del times # clean up memory
        load_file = Path('main.npz')
        if load_file.is_file():
            loaded = np.load('main.npz')
            time_bins = loaded['time_bins']
            mpv_mip = loaded['mpv_mip']
        else:
            time_bins = np.linspace(min_time,max_time, N_TIME_BINS) # 100 bins over 35
            # days  -> ~8 hours per bin
            mpv_mip = np.zeros(len(time_bins)-1)
            iterator = range(len(time_bins)-1)
            if VERBOSE:
                iterator = progressbar.progressbar(iterator)
            for i in iterator: #every 8 hours (roughly)
                # look for rows where timestamp falls inbetween two bin values
                conditions = {'time_lower_bound': time_bins[i],
                              'time_upper_bound': time_bins[i+1]}
                pulseheights = np.array([row['pulseheights']
                                for row in
                                events.where('(timestamp>=time_lower_bound) & '
                                            '(timestamp<time_upper_bound)',
                                             condvars=conditions)]).flatten()
                if len(pulseheights)>0:
                    # create histogram and pass this to the FindMostProbableValueInSpectrum
                    # class from Sapphire, which finds the mpv in the pulseheight spectrum
                    # by fitting a normal distribution
                    h, bins = np.histogram(pulseheights, bins=np.linspace(0, 2000, 50))
                    findMPV = FindMostProbableValueInSpectrum(h,bins)
                    mpv, found = findMPV.find_mpv()
                    if found:
                        mpv_mip[i] = mpv
            np.savez('main.npz', mpv_mip=mpv_mip, time_bins=time_bins)

        count = 0
        ignored = 0
        if VERBOSE:
            iterator = progressbar.progressbar(events.iterrows(start=0, stop=entries,
                                                               step=steps),
                                               max_value=entries, redirect_stdout=True)
        else:
            iterator = events.iterrows(start=0, stop=entries, step=steps)
        # loop over all rows
        for row in iterator:
            if count == 49783: # this event is wrong so hardcode it out YOLO
                continue
            traces = row['traces']
            timings_ADC = []
            padded = row['traces_padded']
            baseline = np.zeros(4)
            for q, trace in enumerate(traces.T):
                if padded[q]: # if the trace is padded, remove the padding
                    trace = trace[1200:]
                if np.max(trace) < ADC_THRESHOLD: # if trace is not above treshold
                    # ignore trace
                    timings_ADC.append(-999)
                else:
                    # first determine the baseline using the first 100 units of the trace
                    t = trace[:100]
                    baseline[q] = np.mean(t)

                    trace = trace - baseline[q]  # normalise trace w.r.t. baseline
                    trigger_point = -999
                    for i, t in enumerate(trace): # find the first value over the treshold
                        if t >= ADC_THRESHOLD:
                            trigger_point = i
                            break
                    timings_ADC.append(trigger_point)

            timings_ADC = np.array(timings_ADC)
            timings_ns = timings_ADC * ADC_TIME_PER_SAMPLE
            timings_ns[timings_ADC<0] = 0
            traces_new = np.empty((4,80))
            pulseheights_event = np.empty(4)
            if np.count_nonzero(timings_ADC<0)>1:
                continue
            if ~(timings_ADC<0).all() and (padded.all() or (~padded).all()): #sometimes
                # only half of the traces are padded, so ignore those. Also ignore
                # cases where there is no trigger at all (there are probably none,
                # but you never know)
                trigger_point = np.min(np.extract(timings_ADC != -999,
                                                  timings_ADC)) # find the first
                                                                # trigger point of the 4 traces
                if trigger_point<5:
                    continue
                if PLOT: # plot all 4 traces in one figure
                    plt.figure()
                if VERBOSE:
                    #print("Row: %s" % row['id'])
                    pass
                for q, trace in enumerate(traces.T):
                    if padded[q]: # remove padding again
                        trace = trace[1200:]
                    # resize trace based on first trigger
                    trace = trace[trigger_point - 5:trigger_point + 75] - baseline[q]
                    if PLOT:
                        plt.plot(get_time_in_ns(trace),trace)
                    if VERBOSE:
                        #print(trace)
                        pass
                    traces_new[q,:] = trace*0.57 # convert ADC count to mV
                if PLOT:
                    plt.savefig('./plots/traces_%s' % count)
                    plt.close()

                # using the timestamp find the time window where this event sits and
                # use the mip peak value from that time window
                timestamp = row['timestamp']
                pos_in_bins = find_closest(time_bins,timestamp)
                mip_peak = mpv_mip[pos_in_bins]

                # read from Kascade.h5
                reconstructed_zenith = row['reconstructed_theta']
                reconstructed_azimuth = row['reconstructed_phi']
                reference_zenith = row['reference_theta']
                reference_azimuth = row['reference_phi']


                timings_event = np.array([row['t1'], row['t2'], row['t3'], row['t4']])
                pulseheights_event = row['pulseheights'] * 0.57 # ADC counts
                #  -> mV -> normalise by mpv
                integrals_event = np.sum(traces_new, axis=1)

                # convert to x,y,z
                x, y, z = azimuth_zenith_to_cartestian(reference_zenith,
                                                       reference_azimuth)

                # save everything
                new_row['id'] = count
                new_row['traces'] = traces_new / mip_peak
                new_row['labels'] = [x,y,z]
                new_row['pulseheights'] = pulseheights_event
                new_row['rec_z'] = reconstructed_zenith
                new_row['rec_a'] = reconstructed_azimuth
                new_row['integrals'] = integrals_event
                new_row['zenith'] = reference_zenith
                new_row['azimuth'] = reference_azimuth
                new_row['timestamp'] = timestamp
                new_row['core_distance'] = row['r']
                new_row['mips'] = pulseheights_event / mip_peak / .57 # because
                # mip peaks were determined using the ADC counts, and we have now
                # converted the pulseheights to mV
                new_row['mpv'] = mip_peak
                new_row['energy'] = row['k_energy']
                new_row['timings'] = timings_event
                new_row.append()
                count += 1
            else:
                if VERBOSE:
                    print('Ignoring entry')
                    print(padded)
                    print(timings_ADC)
                ignored +=1
        if VERBOSE:
            print('Entries ignored: %s' % ignored)
            print('Total new entries: %s' % count)

        # save the time_bins and the mips per bin, for future reference
        table.attrs.time_bins = time_bins
        table.attrs.mips_per_time_bin = mpv_mip
        table.flush()