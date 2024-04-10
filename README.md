# filopodium-spine-learning

This repository contains the code to reproduce results shown in the paper Learning with Filopodia and Spines (Albesa-González and Clopath, 2023). The paper presents a model of spike-timing-dependent plasticity (STDP) where synapses can follow strongly or weakly competitive dynamics depending on their synaptic state (Filopodium-Spine STDP, FS-STDP).

## Repoistory Structure

The repository is structured as follows:

- Figure_X_data.py runs the simulations necessary to producte Figure X and Supplementary Figures associated to Figure X, and stores the results it the Data folder.

- Figure_X.py reads the generated data and creates the panels of Figure X (and associated supplementary figures) as shown in the paper.

- run_network-functions.py contains the actual model, which is implemented using BRIAN2.

- filopodium_spine_learning_notebook.ipynb contains a notebook that allows easy interaction with the model, for those interested in playing around with it!
