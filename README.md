# Slice-Level Attention Module (SLAM)
This repository contains the code for my final degree project in Biomedical Engineering. The project focuses on breast cancer detection in digital breast tomosynthesis (DBT) images using deep learning, with the goal of reducing computational resource usage while achieving equal or better diagnostic performance.

The baseline architecture used is Faster R-CNN with a ResNet-50 backbone. The feature maps from the final layer of ResNet-50 are passed through a simple attention module that calculates slice-level weights. These weights are then used to aggregate the feature maps from the last three ResNet layers across all slices, resulting in a single representative feature map per layer. These aggregated maps are subsequently processed through a Feature Pyramid Network (FPN), preparing them as input for the Faster R-CNN detector.

The model was trained using Google Colab Pro with a TPU v2–8. The results indicate some degree of overfitting. Despite the results, the proposed Slice-Level Attention Module could serve as a foundation for future improvements, such as using more advanced attention mechanisms or training on higher-performance hardware, which would allow to reject the overfitting hypothesis and continue building a more solid foundation from this proposal.

Also it has been developed a way to generate synthetic mammograms from a normalized sigmoid function to weight each slice, which will be combined with a maximum intensity projection (MIP) to generate it. Subsequently, the range of intensities will be modified with different window values and window levels. This can be found in the extraFunctions/SMgenerator.py.

To understand the applicability of this project, a small interface simulating a tomosynthesis viewer has been developed in the Application folder. To use it, it would first be necessary to have the weights obtained from the training of the model and the images of the interface itself. If these are required, do not hesitate to contact me.

For further details, the full final degree project report (in Spanish) is also included in this repository, in the file TFG.pdf. Aslo, the differents Google Colab notebooks can be found in this repository to replicate the project if needed.
