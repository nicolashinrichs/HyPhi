# hyphi – **data**

    Last update:    May 19th, 2026

***

Small data files (< 100 MB) used by the test suite, tutorials, and worked examples live in
this directory.  Larger DuoRhythm files are hosted on OSF (link below) and should be
symlinked in (`ln -s SOURCE_FOLDER ./data`) rather than copied.

## Description of data

This directory houses the smaller (< 100 MB) data sets that are a result of simulations and experiments. Larger files from the DuoRhythm dataset will be published in a data descriptor.

`connectome/{1-8}_connectome_kuramoto.pkl` contains a list of 24 (timepoints based on sliding window) `networkX` objects, each with shape (152 x 152), representing the PLV values for each oscillator pair in a virtual coupled-brain.
Same structure applies for `connectome/avg_connectome_kuramoto.pkl`, which contains PLV matrices averaged over 8 simulations (as per no. of resting states data).

Also, these simulations are done with coupling strength `c_intra` of 25 and `c_inter` of 10**-4, given the RS simulation showed increasing divergence from shuffled starting from around 20-30 `c_intra`.

## Preprocessing

The shipped pickles are direct outputs of the connectome-informed Kuramoto simulator in
`hyphi.simulation` (no further preprocessing).  

## COPYRIGHT/LICENSE

Data files are released under the same BSD-3-Clause licence as the source (see top-level
`LICENSE`).  The DuoRhythm dataset on OSF is governed by its own data-use agreement; consult
the OSF page before redistributing.
