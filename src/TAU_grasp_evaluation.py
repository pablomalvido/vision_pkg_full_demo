class TAUGraspEvaluation():
    '''
    Implements the grasp evaluation. Determines if a cables passes through the grasping area or not
    '''

    def __init__(self, grasping_point, grasp_area):
        self.grasping_point = grasping_point
        self.grasp_area = grasp_area
        print(grasping_point)
        print(grasp_area)

    def exec(self, cable_points):
        grasped = False
        for point in cable_points:
            if (point[0] >= self.grasping_point[0] - int(self.grasp_area[0]/2)) and (point[0] <= self.grasping_point[0] + int(self.grasp_area[0]/2)) and (point[1] >= self.grasping_point[1] - int(self.grasp_area[1]/2)) and (point[1] <= self.grasping_point[1] + int(self.grasp_area[1]/2)):
                grasped = True
                break
        return grasped