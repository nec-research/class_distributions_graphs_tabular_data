#!/bin/bash

pydgn-train --config-file MODEL_CONFIGS/mlp_abalone.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_adult.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS/mlp_cora.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_drybean.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_waveform.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_musk.yml
pydgn-train --config-file MODEL_CONFIGS/mlp_isolet.yml

pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_abalone.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_adult.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS_COSINE/mlp_cora.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_drybean.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_waveform.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_musk.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/mlp_isolet.yml
