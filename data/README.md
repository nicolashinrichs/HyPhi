## `data`
This directory houses the smaller (< 100 MB) data sets that are a result of simulations. Larger files (i.e., the DuoRhythm dataset) are stored in the projects' OSF repository: https://osf.io/yah5u/

{1-8}_connectome_kuramoto.pkl contains a list of 24 (timepoints based on sliding window) networkX objects, each with shape (152 x 152), representing the PLV values for each oscillator pair in a virtual coupled-brain. Same structure applies for avg_connectome_kuramoto.pkl, which contains PLV matrices averaged over 8 simulations (as per no. of resting states data).

Also, these simulations are done with coupling strength c_intra of 25 and c_inter of 10**-4, given the RS simulation showed increasing divergence from shuffled starting from around 20-30 c_intra.
