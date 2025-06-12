# Slice-Level Attention Module (SLAM)
This repository contains the code for my final degree project in Biomedical Engineering. The project focuses on breast cancer detection in digital breast tomosynthesis (DBT) images using deep learning, with the goal of reducing computational resource usage while achieving equal or better diagnostic performance.

The baseline architecture used is Faster R-CNN with a ResNet-50 backbone. The feature maps from the final layer of ResNet-50 are passed through a simple attention module that calculates slice-level weights. These weights are then used to aggregate the feature maps from the last three ResNet layers across all slices, resulting in a single representative feature map per layer. These aggregated maps are subsequently processed through a Feature Pyramid Network (FPN), preparing them as input for the Faster R-CNN detector.

The model was trained using Google Colab Pro with a TPU v2–8. The results indicate some degree of overfitting. However, the proposed Slice-Level Attention Module could serve as a foundation for future improvements—such as using more advanced attention mechanisms or training on higher-performance hardware.

For further details, the full final degree project report is also included in this repository, in the file TFG.pdf.
