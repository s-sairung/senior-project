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

        [STATIC VALUE]  id:                 Is a label of bbox (from object detection)                          
                        scale:              Is an area of bbox
                        frames:             Is a frames of the object that got detected (Form the footage)
        [STATIC VALUE]  frames_cap:        Is a maximum number of frames that will be stored in the object     
        [STATIC VALUE]  frames_threshold:   Is a minimum number of frames that required for prediction          
                        trajectories:       Is a queue that store the bbox centriods (world coordinates from real picture) for the last [frame_size] prior
                                            (I use queue becuase we can dequeue the oldest one out then put newest in for updated prediction)
                        times_tracked:      Count the times that this object get tracked 
    '''

    
    # ----- Creating new object with structure like above -----
    

    def __init__(self, bbox, frame):
        
        worldXcentroid = round((bbox[0] + bbox[2])/2, 4)
        worldYcentroid = round((bbox[1] + bbox[3])/2, 4)
        centroidarr = [worldXcentroid, worldYcentroid]
        self.frames_cap = 30                                        #[EDIT HERE] 

        self.id = bbox[-1]
        self.x = abs(round(bbox[0] - bbox[2], 4))
        self.y = abs(round(bbox[1] - bbox[3], 4))
        self.scales = []
        self.scales.append(round(self.x * self.y, 4))

        self.frames = []
        self.frames.append(frame)
        self.frames_threshold = 30                                  #[EDIT HERE]

        self.trajectories = []
        self.trajectories.append(centroidarr)
        self.times_tracked = 1

        '''
            ========= [Debugging Section] ======== [1/1]
        
        print("Object Initialized.")
        ic([self.id, self.times_tracked, self.x, self.y])
        ic(self.frames)
        ic(self.trajectories)
        ic(self.scales)

        #    ====== [End of Debugging Section] =====
        '''

    '''
        ----- Updating the object with new information -----
        With condition of whether the object has reached its capacity or not
        
        If Yes, it will overwrite the oldest data with new one
        If No, it will just store the new one

    '''

    def update(self, new_bbox, frame):

        # Calculating the data like what it did when initializing
        newWorldXcentroid = round((new_bbox[0] + new_bbox[2])/2, 4)
        newWorldYcentroid = round((new_bbox[1] + new_bbox[3])/2, 4)
        newCentroidarr = [newWorldXcentroid, newWorldYcentroid]

        new_x = abs(round(new_bbox[0] - new_bbox[2], 4))
        new_y = abs(round(new_bbox[1] - new_bbox[3], 4))
        new_scale = round(new_x * new_y, 4)

        '''
            ========= [Debugging Section] ======== [1/2]
        
        print("Before update")
        ic([self.id, self.times_tracked, self.x, self.y])
        ic(self.frames)
        ic(self.trajectories)
        ic(self.scales)
        '''
        #    ====== [End of Debugging Section] =====
        

        self.x = new_x
        self.y = new_y

        #Condition events If -> add new, else -> overwrite
        if(self.times_tracked < self.frames_cap):
            self.scales.append(new_scale)
            self.frames.append(frame)
            self.trajectories.append(newCentroidarr)
            self.times_tracked += 1
        else:
            self.scales.pop(0)
            self.scales.append(new_scale)
            self.frames.pop(0)
            self.frames.append(frame)
            self.trajectories.pop(0)
            self.trajectories.append(newCentroidarr)
            self.times_tracked += 1

        '''
            ========= [Debugging Section] ======== [2/2]
        
        print("After update")
        ic([self.id, self.times_tracked, self.x, self.y])
        ic(self.frames)
        ic(self.trajectories)
        ic(self.scales)
        '''
        #    ====== [End of Debugging Section] =====
            
    '''
        ----- Predicting the object whereabout from its stored informations -----
        With condition of whether the object has reached the minimum threshold or not
        
        If Yes, it will make a new linear model with current sets of values,
                and then input the number of frames ahead [PARAMETER VALUE] for prediction
        If No, it will return value of -1

    '''

    def predict_ahead(self, frames_ahead):   

        if(self.times_tracked < self.frames_threshold):
            return -1
        
        #Since linear regression can only output one value at a time, so I need to decompose the coordinates into x and y
        cen_x = []
        cen_y = []

        trajectories = self.trajectories
        for traj in trajectories:
            cen_x.append(traj[0])
            cen_y.append(traj[1])

        #Scikit use numpy structure, so I'll change it into numpy.
        frames, cen_x, cen_y, scales = np.array(self.frames).reshape(-1,1), np.array(cen_x), np.array(cen_y), np.array(self.scales)

        #Constructing with existed data model by using frame number as regressor and x, y, scale as respond
        model_centroid_x = LinearRegression().fit(frames, cen_x)
        model_centroid_y = LinearRegression().fit(frames, cen_y)

        model_scale = LinearRegression().fit(frames, scales)

        # coefficient of determination
        r_sq_centroid_x = model_centroid_x.score(frames, cen_x)
        r_sq_centroid_y = model_centroid_y.score(frames, cen_y)
        r_sq_scale = model_scale.score(frames, scales)

        # predictions
        pred_frame = np.array(frames[-1] + frames_ahead).reshape(1, -1)

        centroid_x_pred = model_centroid_x.predict(pred_frame)
        centroid_y_pred = model_centroid_y.predict(pred_frame)
        scale_pred = model_scale.predict(pred_frame)

        centroid_x_pred = round(centroid_x_pred[0], 4)
        centroid_y_pred = round(centroid_y_pred[0], 4)
        scale_pred = round(scale_pred[0], 4)

        #ic([cen_x[-1], cen_y[-1]])

        delta_x = round(centroid_x_pred - cen_x[-1], 4)
        delta_y = round(centroid_y_pred - cen_y[-1], 4)

        pred_coordinate = [self.x + delta_x, self.y + delta_y]
        pred_centroid = [centroid_x_pred, centroid_y_pred]

        delta_scale = round(scale_pred - self.scales[-1], 4)
        prediction = [self.id, self.frames[-1] + frames_ahead, pred_coordinate, pred_centroid, delta_scale]

        '''
            ========= [Debugging Section] ======== [1/1]
        '''
        print("Predict result")
        ic(self.id, self.frames[-1], self.frames[-1] + frames_ahead)
        ic([cen_x[-1], cen_y[-1]], [centroid_x_pred, centroid_y_pred])
        ic(self.scales[-1], scale_pred, delta_scale)
        
        #    ====== [End of Debugging Section] =====

        return (prediction)