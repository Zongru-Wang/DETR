# Before you review the code
About 40% of this code base fpr model traning is based on a code example from HuggingFace Inc.

Copyright 2024 The HuggingFace Inc. team. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
# Project Despription
1. This project use a pre-trained detr-resnet-50 model: DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images).
   It was introduced in the paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) by Carion et al.
   and first released in [this repository](https://github.com/facebookresearch/detr).
   You can use the raw model for object detection. See the [model hub](https://huggingface.co/models?search=facebook/detr) to look for all available DETR models.
2. I have not upload the image data to the GitHub
3. The ./Single-Images-With-Label contains images and labels for object detection by using the "labeling" package in python with .txt label format.The Images are PRPD,
   If you need more information for PRPD, please check the README.md file in another repo of mine: (https://github.com/Zongru-Wang/Final-Vit). The experts marked the location
   and the type of the PDs in PRPD pattern.  
5.  ./SJJ_MIX/data-maker.py is used to combine the single-source PRPD into Multi-Source PRPD, which apply random shift for single PRPDs to make Mutlti-source PDs for testing.
     This dataset is aim for testing the model ability for PD detection when PD patterns overlap with each other and when noise signal overlaps the PD singals in PRPDs. This script will combine both the
    Images and labelings together.
7. To-Coco-converter.py is to convert the .txt style labeling into COCO annotation to train the detr-resnet model, the output is ./coco_annotation.json, which contains the image name,path, object
   locations, class informations.
8. Custom_coco.py is used for the training, that transfer the coco_annotations into class ojects during the training .
9. Use .from_pretrained("./facebook/detr-resnet-50") rather than .from_pretrained("./detr-resnet-50") to loading the model unless you download the model from their site.
10. Training-with-AMD.py is designed for AMD graphic card.
    use
    python  Training-with-AMD.py --annotation_file ./coco_annotations.json --img_dir ./Single-Images-With-Label  --num_train_epochs 300 --per_device_train_batch_size 4 --per_device_
eval_batch_size 4 --learning_rate 5e-5 --ignore_mismatched_sizes --image_square_size 256 --checkpointing_steps epoch --output_dir "detr-resnet-50-finetuned" --report_to tensorboardd
    to start training
11. Training-with-cuda.py is designed for graphic card to Cuda.
    accelerate launch Training-with-cuda.py --annotation_file ./coco_annotations.json --img_dir ./Single-Images-With-Label  --num_train_epochs 300 --per_device_train_batch_size 4 --per_device_
eval_batch_size 4 --learning_rate 5e-5 --ignore_mismatched_sizes --image_square_size 256 --checkpointing_steps epoch --output_dir "detr-resnet-50-finetuned" --report_to tensorboardd --resume_from_checkpoint de
tr-resnet-50-finetuned/epoch_29
 use
accelerate --config
to configure your environment.
12. This repo might be set to private at anytime.

  
