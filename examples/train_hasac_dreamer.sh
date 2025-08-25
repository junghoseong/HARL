#!/bin/bash

# Train HASAC + Dreamer hybrid on SMAC 3m
python train_hasac_dreamer.py --env smac --map_name 3m --dreamer_train_interval 10 --imagination_ratio 0.5

# Alternative configurations:
# python train_hasac_dreamer.py --env smac --map_name 8m --dreamer_train_interval 5 --imagination_ratio 0.3
# python train_hasac_dreamer.py --env pettingzoo_mpe --scenario simple_spread_v2 --dreamer_train_interval 15 --imagination_ratio 0.7 