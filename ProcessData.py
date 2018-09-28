import tables
import h5py
import numpy as np
import matplotlib
try:
    cfg = get_ipython().config
    if cfg['IPKernelApp']['kernel_class'] == 'google.colab._kernel.Kernel':
        pass
    else:
        matplotlib.use('Agg')
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from progressbar import progressbar

def process_kascade(path_to_file, new_h5, trigger = 3, verbose=True, std_timings=None,
                    logtransform=True):
    """
    This is just a simple function which takes the h5 file produced by ProcessRawData.py and
    applies the log transformations and saves it to a h5 file Keras can read.
    :param path_to_file: orginal file created by ProcessRawData.py
    :param new_h5: output
    :param trigger: minimum number of triggered stations
    :param verbose: if True print some stuff
    :param std_timings: you need to manually enter the standard deviation of the
    timings in order to get the correct result. For this data I used 10.26,
    this rescales the timings in such a way that they are from -1 to 1. Because of
    outliers just calling np.std gives a very high std and so the data gets
    resclaled badly.
    :param logtransform: if True log transform the traces
    :return: nothing
    """
    with tables.open_file(path_to_file, 'r') as data:
        events = data.root.kascade.events
        entries = len(events)
        with h5py.File(new_h5, 'w') as f:
            traces = f.create_dataset('traces', shape=(entries, 4, 80),
                                      dtype='float32', chunks=True)
            labels = f.create_dataset('labels', shape=(entries, 3), chunks=True)
            input_features = f.create_dataset('input_features',
                                              shape=(entries, 4, 2),
                                              dtype='float32', chunks=True)
            rec_z = f.create_dataset('rec_z', shape=(entries,),
                                     dtype='float32', chunks=True)
            rec_a = f.create_dataset('rec_a', shape=(entries,),
                                     dtype='float32', chunks=True)
            zenith = f.create_dataset('zenith', shape=(entries,),
                                      dtype='float32', chunks=True)
            azimuth = f.create_dataset('azimuth', shape=(entries,),
                                       dtype='float32', chunks=True)
            energy = f.create_dataset('energies', shape=(entries,),
                                      dtype='float32', chunks=True)
            core_distance = f.create_dataset('core_distance', shape=(entries,),
                                      dtype='float32', chunks=True)
            total_traces = np.zeros(shape=(entries,4))
            timings = np.zeros(shape=(entries,4))
            i = 0
            for row in progressbar(events.iterrows(), max_value=entries):
                if np.count_nonzero(np.isnan(row['timings'])) <= (4-trigger):
                    t = row['traces']
                    if logtransform:
                        t = np.log10(t + 1)

                    traces[i,] = t
                    labels[i,] = row['labels']
                    rec_z[i] = row['rec_z']
                    rec_a[i] = row['rec_a']
                    core_distance[i] = row['core_distance']
                    energy[i] = row['energy']
                    zenith[i] = row['zenith']
                    azimuth[i] = row['azimuth']
                    total_traces[i,] = np.sum(row['traces'], axis=1)
                    timings[i,] = row['timings']

                    i+=1
            traces.resize(i, axis=0)
            labels.resize(i, axis=0)
            input_features.resize(i, axis=0)
            rec_z.resize(i, axis=0)
            rec_a.resize(i, axis=0)
            zenith.resize(i, axis=0)
            azimuth.resize(i, axis=0)
            energy.resize(i, axis=0)
            core_distance.resize(i, axis=0)
            timings.resize((i,4))
            total_traces.resize((i, 4))

            total_traces = np.log10(total_traces + 1)
            total_traces -= np.mean(total_traces, axis=1)[:, np.newaxis]
            print("Std of integrals: %s" % np.std(total_traces))
            total_traces /= np.std(total_traces)

            # normalize the timings
            idx = timings != 0.
            timings[~idx] = np.nan
            timings -= np.nanmean(timings, axis=1)[:, np.newaxis]
            if verbose:
                plt.figure()
                plt.hist(
                    np.extract(~np.isnan(timings.flatten()), timings.flatten()),
                    bins=np.linspace(-100, 100, 50))
                plt.savefig('histogram_timings.png')
            print('Std of timings: %s' % np.nanstd(timings))
            if std_timings is not None:
                std_timings = np.nanstd(timings)
            timings /= std_timings
            timings[~idx] = 0.
            timings[np.isnan(timings)] = 0

            input_features[:, :, :] = np.stack((timings, total_traces), axis=2)