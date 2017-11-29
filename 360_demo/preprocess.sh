#!/bin/bash


#python selective_search.py
#python selective_search.py
#CUDA_VISIBLE_DEVICES=2 python image_convert.py
CUDA_VISIBLE_DEVICES=2 python image_convert_fast.py
CUDA_VISIBLE_DEVICES=2 python selective_search.py
#CUDA_VISIBLE_DEVICES=2 python 360demo.py
