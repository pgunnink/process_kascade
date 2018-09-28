import tables
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

""""
This code tries to match up events between the simulation and kascade data and plots 
the traces from both
"""

with tables.open_file('/user/pgunnink/public_html/Kascade/main_data_[70001].h5','r') as \
        data_sim:
    with tables.open_file('/user/pgunnink/public_html/Kascade/kascade.h5','r') as \
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
        np.set_printoptions(precision=2)
        while count<total_plot and count<len(sim_found) and count<len(k_found):
            k_timings = k_found[count]['timings'].reshape(4)
            k_timings -= np.min(k_timings)

            sim_timings = sim_found[count]['timings'].reshape(4)
            sim_timings -= np.min(sim_timings)

            print('Kascade timings: {0}'.format(k_timings))
            print('Simulation timings: {0}'.format(sim_timings))
            plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.plot(k_found[count]['traces'].T)
            plt.legend(['1', '2','3', '4'])
            plt.title('Kascade data %s' % k_timings)
            plt.subplot(122)
            plt.plot(-1*np.reshape(sim_found[count]['traces'].T, (80,4)))
            plt.legend(['1', '2','3', '4'])
            plt.title('Simulation %s' % sim_timings)
            plt.savefig('compare%s.png' % count)
            count += 1
