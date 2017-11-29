#!/bin/bash
GPU_ID=2
WEIGHTS=/home/bskim/workspace/visionAndLanguage/natural-language-object-retrieval/exp-referit/caffemodel/scrc_full_vgg_init.caffemodel

/home/user01/caffe/build/tools/caffe train \
    -solver /home/bskim/workspace/visionAndLanguage/natural-language-object-retrieval/prototxt/scrc_full_vgg_solver.prototxt \
    -weights /home/bskim/workspace/visionAndLanguage/natural-language-object-retrieval/exp-referit/caffemodel/scrc_full_vgg_init.caffemodel