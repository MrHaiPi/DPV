# DPV

The rapid growth of global trade heavily depends on efficient passive positioning technologies based on Low Earth Orbit (LEO) constellation. However, the extensive monitoring range of the LEO constellation generates massive data, challenging traditional solutions to achieve high positioning accuracy and speed simultaneously. To efficiently process the massive data, researchers have studied passive positioning supported by deep learning. However, current studies have been one-sided, assuming fixed values for the number of emitters and receivers and only considering partial data for passive positioning. These approaches are not universal and unsuitable for the LEO constellation positioning scene. To address these issues, we first propose an end-to-end Deep neural network for the Passive positioning of emitters and receivers with Variable numbers called DPV. We introduce Zero Padding to address the inconsistency between the variable number of emitters and receivers and the fixed input/output structure of the neural networks. Additionally, we propose Sort Invariant Training to resolve the permutation problem in multi-target output during the training process. Finally, we publish a passive positioning dataset for emitters and receivers with the variable number in a multi-signal aliasing condition. In this dataset, the proposed method surpasses traditional approaches, delivering state-of-the-art performance by simultaneously achieving exceptional positioning accuracy and time levels.


The dataset: https://modelscope.cn/datasets/ruixia/DPV_Passive_Positioning_Dataset/files
