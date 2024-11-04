# An Approach based on Machine Vision for the identification and shape estimation of deformable linear objects


## Abstract

This system is presented in two research articles: 

* *Malvido Fresnillo, P., Vasudevan, S., Mohammed, W. M., Martinez Lastra, J. L., & Perez Garcia, J. A. (2023). An approach based on machine vision for the identification and shape estimation of deformable linear objects. Mechatronics, 96, 103085. doi: 10.1016/j.mechatronics.2023.103085* (https://www.sciencedirect.com/science/article/pii/S0957415823001411): Introduces the computer vision algorithm pipeline.

* *MalvidoFresnillo, P., Mohammed, W. M., Vasudevan, S., PerezGarcia, J. A., & MartinezLastra, J. L. (2024). Generation of realistic synthetic cable images to train deep learning segmentation models. Machine Vision and Applications, 35(4), 1–14. doi: 10.1007/s00138-024-01562-y* (https://link.springer.com/article/10.1007/s00138-024-01562-y): Proposes a methodology to generate photo-realistic synthetic DLO images (and their corresponding segmentation masks) using Blender. This allows to enhance the semantic segmentation module of the system with a U-Net model trained on the synthetic cables dataset.

The proposed approach processes the different DLOs in the image sequentially, repeating the following procedure for each of them. First, the DLO is segmented. Three different segmentation techniques can be used (edges and color analysis (0), edges and color analysis + spurious edges detector (1), U-Net segmentation (2)). Next, the remaining pixels are analyzed using evaluation windows to identify a series of points along the DLO’s skeleton. These points are  then employed to model the DLO’s shape using a polynomial function. Finally, the output is evaluated by an unsupervised self-critique module, which  validates the results, or fine-tunes the system’s parameters and repeats the process.

## Installation

Main dependencies
```
ROS Melodic
python (3.8.0)
opencv-python (4.8.0.74)
numpy (1.24.4)
tensorflow (2.10.0)
keras (2.10.0)
matplotlib (3.7.5)
Pillow (10.0.0)
h5py (3.9.0)
```

## Usage

The system is initialized with the **vision_pkg_full_demo vision.launch** launch file.
The following parameters need to be modified in the **TAU_supervisor.py** file to use the system with new images:

*  `cable_colors`: BGR color codes of the analyzed DLOs

*  `cable_color_order`: Order of the cables from bottom to top

*  `cable_lengths`: Length of the cables in mm from bottom to top

*  `con_dim`: Size of the main wiring harness connector in mm

* `mold_dim`: Size of the mold/fixture for the main wiring harness connector in mm (Optional)

*  `cable_D`: Diameter of the smallest DLO in mm
  

### Services
  
The system provides three main functionalities that can be called with ROS Services:

*  **/vision/estimate_cables_shape_srv:** Estimates the shape of all the cables of the wire harness (Some examples can be seen in the imgs/ and plots/ folders). It has the following parameters:

	*  `img_path`: (String) Absolute path to the analyzed image.
	* `wh_id:` (String) ID of the wire harness to detect, whose structure if defined in TAU_supervisor.py. Currently only '1' is defined.
	* `pixel_D`: (Int) Desired number of pixels per DLO diameter. Increasing this parameter will increase the accuracy but also the computation time. Typically: 5-8.
	* `forward`: (Bool) True to use forward points propagation. False to use backward points propagation. Typically: False
	* `iteration`: (Bool) True to use the unsupervised self-critique module. Typically: False
	* `segm_opt`: (Int) Segmentation options: **0**: color filter + edge detector, **1**: color filter + edge detector + spurious edge removal, **2**: UNet model. Typically: 2
	* `analyzed_length`: (Int) Cable length in mm to be analyzed. Typically: 80-120.
	* `visualize`: (Bool) If True it only shows the cable initial points, if False it estimates the cables shape. Typically: False
	* `connector_lower_corner`: (Int[]) [Y,X] pixel coordinates of the lower connector corner.
	* `connector_upper_corner`: (Int[]) [Y,X] pixel coordinates of the upper connector corner.


*  **/vision/grasp_point_determination_srv:** Computes the optimal grasp point to separate two cable groups from a wire harness (One example can be seen in the imgs/ and plots/ folders). It has the following parameters:

	*  `img_path`: (String) Absolute path to the analyzed image.
	* `wh_id:` (String) ID of the wire harness to detect, whose structure if defined in TAU_supervisor.py. Currently only '1' is defined.
	* `separated_cable_index:` (Int[])
	* `pixel_D`: (Int) Desired number of pixels per DLO diameter. Increasing this parameter will increase the accuracy but also the computation time. Typically: 5-8.
	* `forward`: (Bool) True to use forward points propagation. False to use backward points propagation. Typically: False
	* `iteration`: (Bool) True to use the unsupervised self-critique module. Typically: False
	* `grasp_point_eval_mm`: (Int[]) [Y,X] mm from the top corner of the mold. Y axis UP, X axis FORWARD. Not applicable for the /vision/grasp_point_determination_srv service (leave it as [0,0]).
	* `analyzed_length`: (Int) Cable length in mm to be analyzed. Typically: 80-120.
	* `simplified`: (Bool) If True it only shows the cable initial points, if False it estimates the cables shape. Typically: True
	* `visualize`: (Bool) If True it only shows the cable initial points, if False it estimates the cables shape. Typically: False
	* `connector_lower_corner`: (Int[]) [Y,X] pixel coordinates of the lower connector corner.
	* `connector_upper_corner`: (Int[]) [Y,X] pixel coordinates of the upper connector corner.
	* `mold_lower_corner`: (Int[]) [Y,X] pixel coordinates of the lower connector fixture/mold corner.
	* `mold_upper_corner`: (Int[]) [Y,X] pixel coordinates of the upper connector fixture/mold corner.

*  **/vision/grasp_point_determination_srv:** Evaluates if the desired wire harness cable groups has been successfully separated (One example can be seen in the imgs/ and plots/ folders). It has the same parameters as the /vision/grasp_point_determination_srv service.


### Examples
The **calls.txt** file contains some example calls to test the previous three services with the images in the **test_images/** directory. The results are stored in the **imgs/** and **plots/** directories.
