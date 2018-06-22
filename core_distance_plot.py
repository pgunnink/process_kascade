import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import tables

path_to_kascade = '/data/hisparc/pgunnink/MachineLearning/Kascade/kascade-20080912.h5'

with tables.open_file(path_to_kascade, 'r') as data:
    print(data.root.reconstructions.colnames)
    core_distances = data.root.reconstructions.col('r')
    #core_distances[:,0] -= 78
    #core_distances[:,1] -= 26
    plt.figure()
    #plt.hist((core_distances[:,0]**2+core_distances[:,1]**2)**(0.5), bins=np.linspace(0,
    #                                                                                100,
    #                                                                           30))

    plt.hist(core_distances, bins=np.linspace(0,100, 30))
    plt.savefig('histogram_core_distances.png')
    plt.close()
