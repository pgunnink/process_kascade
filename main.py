#!/data/hisparc/pgunnink/miniconda3/envs/kascade/bin/python
#PBS -q short
#PBS -d /data/hisparc/pgunnink/Kascade/process_kascade

import tables
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import h5py
import progressbar
import pdb
from sapphire.analysis.find_mpv import FindMostProbableValueInSpectrum

from ProcessDataML.DegRad import azimuth_zenith_to_cartestian

path_to_kascade = '/data/hisparc/pgunnink/MachineLearning/Kascade/kascade.h5'
path_to_new_file = '/data/hisparc/pgunnink/MachineLearning/Kascade/ML_file.h5'


ADC_THRESHOLD = 20
ADC_TIME_PER_SAMPLE = 2.5
N_stations = 1
VERBOSE = True
N_TIME_BINS = 3
PLOT = False
overwrite = True
steps = 1

def get_time_in_ns(trace):
    return np.linspace(0,np.max(trace.shape)*ADC_TIME_PER_SAMPLE,np.max(trace.shape))

def find_closest(bins, value):
    for i in range(len(bins)-1):
        if value>=bins[i] and value<bins[i+1]:
            return i



with h5py.File(path_to_new_file, 'w') as f:
    with tables.open_file(path_to_kascade,'r') as data:
        # events with offsets corrected are stored in /reconstructions_offsets
        events = data.root.reconstructions_offsets
        entries = len(events)
        if VERBOSE:
            print("Total entries before filtering: %s" % entries)

        # first create an mpv fit per 8 hours, which is used later to normalize the
        # traces
        times = events.col('timestamp')
        min_time = times.min()
        max_time = times.max()+1
        del times # clean up memory

        time_bins = np.linspace(min_time,max_time, N_TIME_BINS) # 100 bins over 35 days -> ~8
                                                        # hours per bin
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


        # create h5py dataset to populate later
        traces_dataset = f.create_dataset('traces', shape=(entries, N_stations * 4, 80),
                                  dtype='float32', chunks=True)
        labels_dataset = f.create_dataset('labels', shape=(entries, 3), dtype='float32', chunks=True)
        input_features_dataset = f.create_dataset('input_features',shape=(entries, N_stations *
                                                                  4, 2), dtype = 'float32', chunks=True)
        pulseheights_dataset = f.create_dataset('pulseheights', shape=(entries, N_stations *
                                                               4), dtype = 'float32', chunks=True)
        rec_z_dataset = f.create_dataset('rec_z', shape=(entries,), dtype='float32', chunks=True)
        rec_a_dataset = f.create_dataset('rec_a', shape=(entries,), dtype='float32', chunks=True)
        zenith_dataset = f.create_dataset('zenith', shape=(entries,), dtype='float32', chunks=True)
        azimuth_dataset = f.create_dataset('azimuth', shape=(entries,), dtype='float32', chunks=True)
        timestamp_dataset = f.create_dataset('timestamp', shape=(entries,), dtype='int32', chunks=True)
        core_distance_dataset = f.create_dataset('core_distance', shape=(entries,),
                                                 dtype='float32', chunks=True)

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
            if count == 49783: # this event is wrong so hardcode it out #YOLO
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


                timings_event = np.array([row['t1'], row['t2'], row['t3'], row['t4']]) \
                                * 10e9
                pulseheights_event = row['pulseheights'] * 0.57 / mip_peak # ADC counts
                #  -> mV -> normalise by mpv
                integrals_event = row['integrals'] * 0.57 / mip_peak # ADC counts ->
                #  mV -> normalise by mpv

                # convert to x,y,z
                x, y, z = azimuth_zenith_to_cartestian(reference_zenith,
                                                       reference_azimuth)

                # save everything
                traces_dataset[count,:] = traces_new / mip_peak # rescale traces by mip
                #  peak
                labels_dataset[count,:] = [x,y,z]
                input_features_dataset[count, :, 0] = timings_ns
                input_features_dataset[count, :, 1] = integrals_event
                pulseheights_dataset[count,:] = pulseheights_event
                rec_z_dataset[count] = reconstructed_zenith
                rec_a_dataset[count] = reconstructed_azimuth
                zenith_dataset[count] = reference_zenith
                azimuth_dataset[count] = reference_azimuth
                timestamp_dataset[count] = timestamp
                core_distance_dataset[count] = row['r']

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
        f.create_dataset('time_bins', data=time_bins, dtype='float32')
        f.create_dataset('mips_per_time_bin', data=mpv_mip, dtype='float32')

        # rescale datasets to account for ignored events
        traces_dataset.resize(count, axis=0)
        labels_dataset.resize(count, axis=0)
        input_features_dataset.resize(count, axis=0)
        input_features_dataset.resize(count, axis=0)
        pulseheights_dataset.resize(count, axis=0)
        rec_z_dataset.resize(count, axis=0)
        rec_a_dataset.resize(count, axis=0)
        zenith_dataset.resize(count, axis=0)
        azimuth_dataset.resize(count, axis=0)
        timestamp_dataset.resize(count, axis=0)
        core_distance_dataset.resize(count, axis=0)

        # normalize data (for now, in production this would be done on the fly)
        timings = input_features_dataset[:][:, :, 0]
        idx = timings != 0.
        timings[~idx] = np.nan
        timings -= np.nanmean(timings, axis=1)[:, np.newaxis]
        plt.figure()
        plt.hist(np.extract(~np.isnan(timings.flatten()), timings.flatten()),
                 bins=np.linspace(-25, 25, 50))
        plt.savefig('histogram_timings.png')
        print('Std of timings: %s' % np.nanstd(timings))
        timing_std = np.nanstd(timings)
        timings /= np.nanstd(timings)
        timings[~idx] = 0.


        total_traces = np.log10(input_features_dataset[:][:, :, 1] + 1)
        total_traces -= np.mean(total_traces, axis=1)[:, np.newaxis]
        integral_std = np.nanstd(total_traces)
        print("Std of integrals: %s" % np.nanstd(total_traces))
        total_traces /= np.nanstd(total_traces)

        input_features_dataset[:] = np.stack((timings,total_traces),axis=2)
        input_features_dataset.attrs['timings_std'] = timing_std
        input_features_dataset.attrs['integrals_std'] = integral_std