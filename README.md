# An Approach based on Machine Vision for the identification and shape estimation of deformable linear objects


### Abstract

This approach processes the different DLOs in the image sequentially, repeating the following procedure for each of them. First, the DLO is segmented by examining the colors and edges  in the image. Next, the remaining pixels are analyzed using evaluation windows to identify a series of points along the DLO’s skeleton. These points are  then employed to model the DLO’s shape using a polynomial function. Finally, the output is evaluated by an unsupervised self-critique module, which  validates the results, or fine-tunes the system’s parameters and repeats the process.

### Installation

Main dependencies
```
python (3.9)
opencv 
numpy
```

### Usage

The following parameters need to be modified in the TAU_supervisor.py file to use the system with new images:

* `img_path`: Path to the analyzed image
* `cable_colors`: BGR color codes of the analyzed DLOs
* `cable_color_order`: Order of the cables from bottom to top
* `con_corner_below`: Position of the lower corner of the wiring harness connector
* `con_corner_above`: Position of the upper corner of the wiring harness connector
* `con_dim`: Size of the wiring harness connector in mm
* `cable_D`: Diameter of the smallest DLO in mm

Additionally, the following parameters can be modified to use different system functionalities:
* `pixel_D`: Desired number of pixels per DLO diameter
* `estimate_dlos`: If False, just the connector corner pixels are visualized
* `forward`: True to use forward points propagation. False to use backward points propagation
* `iteration`: True to use the unsupervised self-critique module

The system can be run by executing the TAU_supervisor.py file. As an example, the system parameters are already configured to estimate the shape of the DLOs in the wh1_pinkblue.jpeg image using the BWP algorithm. Therefore, this file can be executed after downloading for testing the package.
