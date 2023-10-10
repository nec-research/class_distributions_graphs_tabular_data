#!/bin/bash

pydgn-train --config-file MODEL_CONFIGS/gin_abalone.yml
pydgn-train --config-file MODEL_CONFIGS/gin_adult.yml
pydgn-train --config-file MODEL_CONFIGS/gin_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS/gin_cora.yml
pydgn-train --config-file MODEL_CONFIGS/gin_drybean.yml
pydgn-train --config-file MODEL_CONFIGS/gin_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS/gin_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS/gin_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS/gin_waveform.yml
pydgn-train --config-file MODEL_CONFIGS/gin_musk.yml
pydgn-train --config-file MODEL_CONFIGS/gin_isolet.yml

pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_abalone.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_adult.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_citeseer.yml
pydgn-train --config-fnhtole MODEL_CONFIGS_COSINE/gin_cora.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_drybean.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_electricalgrid.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_occupancy.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_pubmed.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_waveform.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_musk.yml
pydgn-train --config-file MODEL_CONFIGS_COSINE/gin_isolet.yml

