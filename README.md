# On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data (NeurIPS 2023)

## Reference

This research software is provided as is. If you happen to use or modify this code, please remember to cite the paper:

    @inproceedings{errica_class_2023,
     author = {Errica, Federico},
     title = {On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data},
     booktitle = {Advances in Neural Information Processing Systems},
     volume = {36},
     year = {2023}
    }

## How to reproduce

The steps below are used to reproduce the experiments of the paper.

### Step 1) Install required packages
We assume Pytorch and Pytorch Geometric >= 2.3.0 are already installed. Then run

    pip install -r requirements.txt

### Step 2) Create dataset

You can prepare the datasets using the following command

    pydgn-dataset --config-file DATA_CONFIGS/config_abalone.yml

(and similarly for the other datasets using the configuration files in the `DATA_CONFIGS` folder.)

### Step 3) Launch all node classification experiments 

Make sure you configure your hardware requirements in the configuration files present in the `MODEL_CONFIGS` folder.
Then you can run

    source launch_mlp_exp.sh
    source launch_simpledgn_exp.sh    
    source launch_gin_exp.sh    
    source launch_gcn_exp.sh    

### Step 4) Launch the synthetic experiments 

    python launch_class_separator_exp.py

This will store some checkpoints that you can load in the notebooks.

### Optional Step) Launch an individual experiment (remove [--debug] to parallelize as in step 3)

    pydgn-train --config-file MODEL_CONFIGS/pedalme/mlp_abalone.yml --debug

This will launch model selection and risk assessment for the MLP and compute the final scores. You can use different
configuration files to launch different experiments.

## Jupyter Notebooks

You can use the jupyter notebooks to inspect our qualitative experiments and animations for the theoretical results.