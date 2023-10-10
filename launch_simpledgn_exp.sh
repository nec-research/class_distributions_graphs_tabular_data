#!/bin/bash

pydgn-train --config-file MODEL_CONFIGS/simpledgn_abalone.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_adult.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS/simpledgn_cora.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_drybean.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_waveform.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_musk.yml
pydgn-train --config-file MODEL_CONFIGS/simpledgn_isolet.yml

pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_abalone.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_adult.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS_COSINE/simpledgn_cora.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_drybean.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_waveform.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_musk.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/simpledgn_isolet.yml
