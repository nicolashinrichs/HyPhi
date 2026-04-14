# hyphi – **data**

    Last update:    April 10, 2026
    Status:         work in progress

***

*Project data should lie in the `data/` folders.
If your research data resides somewhere else and is too big to copy, you can easily create [symbolic links](https://stackoverflow.com/questions/1951742/how-can-i-symlink-a-file-in-linux) from these sources to your `data/` folder in your project structure (`ln -s SOURCE_FOLDER ./data`).*

## Description of data

This directory houses the smaller (< 100 MB) data sets that are a result of simulations and experiments. Larger files from the DuoRhythm dataset are stored in the projects' OSF repository: https://osf.io/yah5u/

`connectome/{1-8}_connectome_kuramoto.pkl` contains a list of 24 (timepoints based on sliding window) `networkX` objects, each with shape (152 x 152), representing the PLV values for each oscillator pair in a virtual coupled-brain.
Same structure applies for `connectome/avg_connectome_kuramoto.pkl`, which contains PLV matrices averaged over 8 simulations (as per no. of resting states data).

Also, these simulations are done with coupling strength `c_intra` of 25 and `c_inter` of 10**-4, given the RS simulation showed increasing divergence from shuffled starting from around 20-30 `c_intra`.

## Preprocessing

*Which preprocessing steps were done, and why. List references and toolboxes (+version), & point to the corresponding code of your project.*

## COPYRIGHT/LICENSE

*One could add information about data sharing, license and copy right issues*
