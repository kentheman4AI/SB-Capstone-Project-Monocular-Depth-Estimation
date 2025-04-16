# SB-Capstone-Project-Monocular-Depth-Estimation
This is the Capstone project for the Springboard Maryland Global Campus Capstone project - Monocular Depth Estimation (depth perception)

# This project fine-tuned MDE models provided by: Depth Anything V2 for Metric Depth Estimation
# Citations for Depth Anything v2 are below

## Citations

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

User Instructions for "Depth Estimation Web Application"
1. Choose image to process
*image to be processed will be displayed

2. Select depth inference to use a pre-trained or fine-tuned model

3. Select the augmented feature as Mirror, Indoor Art, or Outdoor Art
*  Mirror and Indoor Art uses an indoor MDE model trained/fine-tuned on a Hypersim dataset and assumes
a maximum depth of 20 meters, while Outdoor Art uses an outdoor MDE model trained/fine-tuned on a Virtual KITTI 2 dataset
and assumes a maximum depth of 80 meters

4. Select grayscale or colormap for output

5. Click to process depth map

6. Image of depth map

7. Links to resources
