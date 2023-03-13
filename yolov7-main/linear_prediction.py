from icecream import ic
import numpy as np
from sklearn.linear_model import LinearRegression

class PredictionBox(object):

    '''
        Object will be represent in this way:

                 _______________________________  (x, y)      
                |                               |
                |                               |
                |                               |
                |                               |
                |               +               |
                |                               |
                |                               |
                |                               |
                |_______________________________|
            (0, 0)    

        id:             label of the bbox
        scale:          Area of bbox
        frames:         Frames of the object that got detected
        trajectories:   Is an queue that store the bbox centriod (world coordinates from real picture) for the last 30 frame prior
                        (I use queue becuase we can dequeue the oldest one out then put newest in for updated prediction)
        times_tracked:  Count the times that this object get tracked 
        


    '''

    def __init__(self, bbox, frame):
        
        worldXcentroid = (bbox[0] + bbox[2])//2
        worldYcentroid = (bbox[1] + bbox[3])//2
        centroidarr = [worldXcentroid, worldYcentroid]

        self.id = bbox[-1]
        self.x = abs(bbox[0] - bbox[2])
        self.y = abs(bbox[1] - bbox[3])
        self.scales = [30]
        self.scales.append(self.x * self.y)
        self.frames = [30]
        self.frames.append(frame)
        self.trajectories = [30]
        self.trajectories.append(centroidarr)
        self.times_tracked = 1

    def predict_ahead(self, frames_ahead):

        if(self.times_tracked < 30):
            return -1

        frames = self.frames
        coor_x = []
        coor_y = []

        trajectories = self.trajectories
        for traj in trajectories:
            coor_x.append(traj[0])
            coor_y.append(traj[1])

        frames, coor_x, coor_y = np.array(frames), np.array(coor_x), np.array(coor_y)

        model_x = LinearRegression().fit(frames, coor_x)
        model_y = LinearRegression().fit(frames, coor_y)

        # coefficient of determination
        r_sq_x = model_x.score(frames, coor_x)
        r_sq_y = model_y.score(frames, coor_y)

        # predictions
        x_pred = model_x.predict(frames[-1] + frames_ahead)
        y_pred = model_y.predict(frames[-1] + frames_ahead)

        bbox = [0, 0, x_pred, y_pred, self.id]
        return (PredictionBox(bbox, frames[-1] + frames_ahead))

    def update(self, new_bbox, frame):
        newWorldXcentroid = (new_bbox[0] + new_bbox[2])//2
        newWorldYcentroid = (new_bbox[1] + new_bbox[3])//2
        newCentroidarr = [newWorldXcentroid, newWorldYcentroid]

        new_x = abs(new_bbox[0] - new_bbox[2])
        new_y = abs(new_bbox[1] - new_bbox[3])
        new_scale = new_x * new_y

        self.x = new_x
        self.y = new_y
        
        if(self.times_tracked < 30):
            self.scales.append(new_scale)
            self.trajectories.append(newCentroidarr)
            self.times_tracked += 1
            return
        else:
            self.scales.pop(0)
            self.scales.append(new_scale)
            self.frames.pop(0)
            self.frames.append(frame)
            self.trajectories.pop(0)
            self.trajectories.append(newCentroidarr)
            prediction = self.predict_ahead()
            return prediction

