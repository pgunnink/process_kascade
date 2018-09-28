This is a git repository which contains code to work with the Kascade data. Installation is simple

``git clone https://github.com/pgunnink/process_kascade``



`ProcessRawData.py` contains all code for reading the stuff from the Kascade data and puts it in a format `ProcessData.py` can read.



`ProcessData.py`contains the code that transforms the main data file created by `ProcessRawData.py`. Just use `from process_kascade.ProcessData import process_kascade` and then call `process_kascade()` from inside the code. See for example the jupyter notebook I put in the HisparcML module.

`compare_data_and_sim.py` contains code for picking events from the Kascade and simulation data that are somewhat similar.

`noise_levels.py` determines the noise levels of Kascade
