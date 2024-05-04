# filopodium-spine-learning

This repository contains the code to reproduce results shown in the paper Learning with Filopodia and Spines (Albesa-González and Clopath, 2023). The paper presents a model of spike-timing-dependent plasticity (STDP) where synapses can follow strongly or weakly competitive dynamics depending on their synaptic state (Filopodium-Spine STDP, FS-STDP).

## Repository Structure

The repository is structured as follows:

- Figure_X_data.py runs the simulations necessary to producte Figure X and Supplementary Figures associated to Figure X, and stores the results in the Data folder.

- Figure_X.py reads the generated data and creates the panels of Figure X (and associated supplementary figures) as shown in the paper.

- run_network_functions.py contains the actual model, which is implemented using BRIAN2.

- filopodium_spine_learning_notebook.ipynb contains a notebook that allows easy interaction with the model, for those interested in playing around with it!

- Note: figure_1_data.py should take a few minutes to run. figure_2_3_data.py and figure_4_data.py perform a computationally expensive parameter sweep, and can take up to a few hours to run using 128 CPU's. To run figure_2_3_data.py and figure_4_data.py use python figure_X_data.py --num_cpu 128 (or any other desired number). Simulations in the notebook are the least computationally expensive and the easiest way to start interacting with the model. 
