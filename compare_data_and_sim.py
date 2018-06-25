import tables
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import progressbar

with tables.open_file('/user/pgunnink/public_html/Kascade/main_data_[70001].h5','r') as \
        data_sim:
    with tables.open_file('/user/pgunnink/public_html/Kascade/Kascade_data.h5','r') as \
            kascade_data:
        kascade_events = kascade_data.root.kascade.events
        sim_events = data_sim.root.traces.Traces
        count = 0
        total_plot = 5
        cd = 30
        zenith_find = 0.35
        azimuth_find = 0.5
        energy_find = 10e15

        options = {'cd': cd,
                   'zenith_find': zenith_find,
                   'azimuth_find': azimuth_find,
                   'energy_find': energy_find}

        filter_kascade = '(zenith<zenith_find+0.2) & (zenith>zenith_find-0.2) & ' \
                         '(azimuth<azimuth_find+.2) & (azimuth>azimuth_find-0.2) & ' \
                         '(core_distance>cd-5) & (core_distance < cd+5) & ' \
                         '(energy>energy_find*0.8) & (energy<energy_find*1.2) '
        k_found = kascade_events.read_where(filter_kascade, options)
        sim_found = sim_events.read_where(filter_kascade, options)
        while count<total_plot and count<len(sim_found) and count<len(k_found):
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.plot(k_found[count]['traces'].T)
            plt.title('Kascade data')
            plt.subplot(122)
            plt.plot(-1*np.reshape(sim_found[count]['traces'].T, (80,4)))
            plt.title('Simulation')
            plt.savefig('compare%s.png' % count)
            count += 1

        timings_kascade = kascade_events.col('timings')/10e9
        timings_sim = sim_events.col('timings').reshape((-1,4))



        idx = timings_kascade != 0.
        timings_kascade[~idx] = np.nan
        idx = list(range(len(timings_kascade)))
        for i, timing in enumerate(timings_kascade):
            if np.count_nonzero(np.isnan(timing)) > 1:
                idx.remove(i)
        timings_kascade = timings_kascade[idx,]

        timings_kascade -= np.nanmean(timings_kascade, axis=1)[:, np.newaxis]
        plt.figure()
        plt.hist(np.extract(~np.isnan(timings_kascade.flatten()), timings_kascade.flatten()),
                 bins=np.linspace(-25, 25, 50))
        plt.savefig('histogram_timings_kascade.png')
        print('Std of Kascade timings: %s' % np.nanstd(timings_kascade))
        print('Total non-nan events: %s' % np.count_nonzero(timings_kascade.flatten()))

        idx = timings_sim != 0.
        timings_sim[~idx] = np.nan
        idx = list(range(len(timings_sim)))
        for i, timing in progressbar.progressbar(enumerate(timings_sim), max_value=len(
                timings_sim)):
            if np.count_nonzero(np.isnan(timing)) > 1:
                idx.remove(i)
        timings_sim = timings_sim[idx,]
        timings_sim -= np.nanmean(timings_sim, axis=1)[:, np.newaxis]
        plt.figure()
        plt.hist(
            np.extract(~np.isnan(timings_sim.flatten()), timings_sim.flatten()),
            bins=np.linspace(-25, 25, 50))
        plt.savefig('histogram_timings_sim.png')
        print('Std of Simulation timings: %s' % np.nanstd(timings_sim))
        print('Total non-nan events: %s' % np.count_nonzero(timings_sim.flatten()))
