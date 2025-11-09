#!/bin/bash

python precompute_valid_pairs.py --base_path /datasets/hida/current/shapegen/images_white_bg --camera_base_path /datasets/hida/current/shapegen/base_images_camera_info --output_path /datasets/hida/current/shapegen/obj_pair_sim_white_bg
python preextract_valid_images.py --unique_images_file /datasets/hida/current/shapegen/obj_pair_sim_white_bg/precomputed_pairs/unique_images.pkl --output_path /datasets/hida/current/shapegen/obj_pair_sim_white_bg
python precomputed_sim.py --valid_pairs_file /datasets/hida/current/shapegen/obj_pair_sim_white_bg/precomputed_pairs/valid_pairs.pkl --features_file /datasets/hida/current/shapegen/obj_pair_sim_white_bg/extracted_features_pairs/all_features.pkl --output_path /datasets/hida/current/shapegen/obj_pair_sim_white_bg