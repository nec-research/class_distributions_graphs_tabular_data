#!/bin/bash

pydgn-train --config-file MODEL_CONFIGS/gcn_abalone.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_adult.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS/gcn_cora.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_drybean.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_waveform.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_musk.yml
pydgn-train --config-file MODEL_CONFIGS/gcn_isolet.yml

pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_abalone.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_adult.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS_COSINE/gcn_cora.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_drybean.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_waveform.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_musk.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gcn_isolet.yml



