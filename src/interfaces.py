from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Tuple

class Segmentation(ABC):
    '''
    Base class for the implementation of segmentation methods.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, input_image: ndarray) -> ndarray:
        pass



class Predictor(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self):
        pass



class MaskProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, mask_image: ndarray):
        pass


class GraphProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, input_image: ndarray, distance_image: ndarray, paths: dict, intersection_points: dict):
        pass


class PathsProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, graph, preds):
        pass


class PathsExcludedProcessing(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, paths, graph, intersections_points):
        pass



class Smoothing(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, paths, key):
        pass



class CrossingLayout(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, splines, paths, nodes, candidate_nodes, input_image, colored_mask):
        pass



class OutputMask(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def exec(self, splines):
        pass


class TAUSegmentationInterface(ABC):
    '''
    Base class for the implementation of TAU segmentation methods.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, color_cable: ndarray, thr1: float, thr2: float, erosion: int) -> Tuple[ndarray, list, int]:
        pass


class TAUPreprocessingInterface(ABC):
    '''
    Base class for the implementation of image preprocessing methods.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self) -> Tuple[ndarray, ndarray, list, list, float, list]:
        pass


class TAUCablePointsEstimationInterface(ABC):
    '''
    Base class for the estimation of the cable points.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, img: ndarray, init: list, window_size: list, last_x: int, n_miss_max: int, evaluate_init: bool, max_deg: int) -> Tuple[list, int, list, bool, int, int]:
        pass


class TAUCableLineEstimationInterface(ABC):
    '''
    Base class for the implementation of the estimation of the cable line with a polynomial regression
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, points: list, step_line: int, limits: list) -> Tuple[list, list]:
        pass


class TAUForwardPropagationInterface(ABC):
    '''
    Base class for the forward propagation. A cable is calculated propagating its pixels forward
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, segm_img: ndarray, segm_pixels: list, initial_point: list, window_size: list, cable_length: float, n_miss_max: int, n_cables: int) -> Tuple[list, int, bool, int, int, bool]:
        pass


class TAUBackwardPropagationInterface(ABC):
    '''
    Base class for the backward propagation. A cable is calculated propagating all the segmented pixels backwards, then we discard small and repeated segments,
     we join the segments that are part of the same line and we select the final line that is closer to the theoretical initial point
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, segm_img: ndarray, segm_pixels: list, initial_point: list, n_cables: int, window_size: list, cable_length: float, init_index: int) -> Tuple[list, int, bool]:
        pass


class TAUCritiqueInterface(ABC):
    '''
    Implements the result evaluation. In case the result is not successful it tunes the system parameters
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass 

    @abstractmethod
    def exec(self, points_cable: list, segm_img: ndarray, n_segm_pixels: int, n_captured_points: int, success_points: bool, n_cables: int, thr1: float, thr2: float, evaluation_window: list, erosion: int, init_success: bool, count_no_borders: int, count_free_steps: int) -> Tuple[bool, int, float, float, int, list]:
        pass