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
                        status:             show the status of the predicted object (0 = nothing in particular, 1 = completely out of bounds, 2 = clipped box)

        [What will this predict]:
            1. Centroid of the object at next 'frames_ahead' frame
            2. Trajectory of the object at next 'frames_ahead' frame 
                (In vector form from current frame to the predicted frame) [REMINDER] X-axis: left-to-right, Y-axis: top-to-bottom
    '''

    
    # ----- Creating new object with structure like above -----
    
    

    def __init__(self, bbox, frame):

        decimal_points = 4             # [EDIT HERE!!]: number of float decimals [1/3]

        worldXcentroid = round((bbox[0] + bbox[2])/2, decimal_points)
        worldYcentroid = round((bbox[1] + bbox[3])/2, decimal_points)
        centroidarr = [worldXcentroid, worldYcentroid]
        self.frames_cap = 30                                        #[EDIT HERE] 

        self.id = bbox[-1]
        self.x = abs(round(bbox[0] - bbox[2], decimal_points))
        self.y = abs(round(bbox[1] - bbox[3], decimal_points))
        self.scales = []
        self.scales.append(round(self.x * self.y, decimal_points))

        self.frames = []
        self.frames.append(frame)
        self.frames_threshold = 30                                  #[EDIT HERE]

        self.trajectories = []
        self.trajectories.append(centroidarr)
        self.times_tracked = 1

        self.status = 0

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
        [Function Descrisption]
        ----- Updating the object with new information -----
        With condition of whether the object has reached its capacity or not
        
        If Yes, it will overwrite the oldest data with new one
        If No, it will just store the new one

    ''' 

    def update(self, new_bbox, frame):

        decimal_points = 4             # [EDIT HERE!!]: number of float decimals [2/3]

        # Calculating the data like what it did when initializing
        newWorldXcentroid = round((new_bbox[0] + new_bbox[2])/2, decimal_points)
        newWorldYcentroid = round((new_bbox[1] + new_bbox[3])/2, decimal_points)
        newCentroidarr = [newWorldXcentroid, newWorldYcentroid]

        new_x = abs(round(new_bbox[0] - new_bbox[2], decimal_points))
        new_y = abs(round(new_bbox[1] - new_bbox[3], decimal_points))
        new_scale = round(new_x * new_y, decimal_points)

        '''
            ========= [Debugging Section] ======== [1/2]
        '''
        print("Before update")
        ic([self.id, self.times_tracked, self.x, self.y])
        ic(self.frames)
        ic(self.trajectories)
        ic(self.scales)
        
        #    ====== [End of Debugging Section] =====
        

        self.x = new_x
        self.y = new_y

        #Condition event: If reach cap, will pop first and then add normally
        if(self.times_tracked >= self.frames_cap):
            self.scales.pop(0)
            self.frames.pop(0)
            self.trajectories.pop(0)

        self.scales.append(new_scale)
        self.frames.append(frame)
        self.trajectories.append(newCentroidarr)
        self.times_tracked += 1

        '''
            ========= [Debugging Section] ======== [2/2]
        '''
        print("After update")
        ic([self.id, self.times_tracked, self.x, self.y])
        ic(self.frames)
        ic(self.trajectories)
        ic(self.scales)
        
        #    ====== [End of Debugging Section] =====
            
    '''
        [Function Descrisption]
        ----- Predicting the object whereabout from its stored informations -----
        With condition of whether the object has reached the minimum threshold or not
        
        If Yes, it will make a new linear model with current sets of values,
                and then input the number of frames ahead [PARAMETER VALUE] for prediction
        If No, it will return value of -1

    '''

    def predict_ahead(self, frames_ahead, video_dimension):

        decimal_points = 4             # [EDIT HERE!!]: number of float decimals [3/3]   

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

        ''' (Currently Useless)
        # coefficient of determination 
        r_sq_centroid_x = model_centroid_x.score(frames, cen_x)
        r_sq_centroid_y = model_centroid_y.score(frames, cen_y)
        r_sq_scale = model_scale.score(frames, scales)
        '''

        # predictions
        pred_frame = np.array(frames[-1] + frames_ahead).reshape(1, -1)

        centroid_x_pred = model_centroid_x.predict(pred_frame)
        centroid_y_pred = model_centroid_y.predict(pred_frame)
        scale_pred = model_scale.predict(pred_frame)                #Predicted Scale by this can be negative which make no sense to me

        centroid_x_pred = round(centroid_x_pred[0], decimal_points)
        centroid_y_pred = round(centroid_y_pred[0], decimal_points)
        scale_pred = round(scale_pred[0], decimal_points)

        # calculate into more meaningful and easier to understand
        current_cen_x = cen_x[-1]
        current_cen_y = cen_y[-1]
        current_scale = self.scales[-1]
        current_frame = self.frames[-1]

        delta_x = round(centroid_x_pred - current_cen_x, decimal_points)
        delta_y = round(centroid_y_pred - current_cen_y, decimal_points)  

        offset_x = self.x + delta_x/2
        offset_y = self.y + delta_y/2

        #Transforming our data into the traditional ones: (x1,y1) at top left and (x2, y2) at bottom right
        pred_x1 = round(centroid_x_pred - offset_x, decimal_points)
        pred_y1 = round(centroid_y_pred - offset_y, decimal_points)
        pred_x2 = round(centroid_x_pred + offset_x, decimal_points)
        pred_y2 = round(centroid_y_pred + offset_y, decimal_points)

        '''
            ========= [Debugging Section] ======== [1/2]
        
        print("First Predict result")
        ic(self.id, current_frame, current_frame + frames_ahead)
        ic([pred_x1, pred_y1, pred_x2, pred_y2], self.status)
        ic([current_cen_x, current_cen_y], [centroid_x_pred, centroid_y_pred])
        ic(current_scale, scale_pred)
        '''
        #    ====== [End of Debugging Section] =====

        # Clip the diagonal line to be within the video boundaries
        clipped = self.line_clip(pred_x1, pred_y1, pred_x2, pred_y2, video_dimension[0], video_dimension[1])
        pred_x1 = clipped[0]
        pred_y1 = clipped[1]
        pred_x2 = clipped[2]
        pred_y2 = clipped[3] 
        self.status = clipped[4]

        if(self.status == 2): #This will also affect the Scale and the Centroid of the object
            dx = abs(pred_x1 - pred_x2)
            dy = abs(pred_y1 - pred_y2)
            sum_x = pred_x1 + pred_x2
            sum_y = pred_y1 + pred_y2           

            scale_pred = round((dx * dy), decimal_points)
            centroid_x_pred = round((sum_x/2), decimal_points)
            centroid_y_pred = round((sum_y/2), decimal_points)

            delta_x = round(centroid_x_pred - current_cen_x, decimal_points)
            delta_y = round(centroid_y_pred - current_cen_y, decimal_points)

        trajectory = [delta_x, delta_y]

        pred_xy = [self.x + delta_x, self.y + delta_y]
        pred_centroid = [centroid_x_pred, centroid_y_pred]

        delta_scale = round(scale_pred - current_scale, decimal_points)

        prediction = [self.id, current_frame + frames_ahead, pred_xy, pred_centroid, trajectory, delta_scale]

        '''
            ========= [Debugging Section] ======== [2/2]
        
        print("Final Predict result")
        ic(self.id, current_frame, current_frame + frames_ahead)
        ic([pred_x1, pred_y1, pred_x2, pred_y2], self.status)
        ic([current_cen_x, current_cen_y], [centroid_x_pred, centroid_y_pred], trajectory)
        ic(current_scale, scale_pred, delta_scale)
        '''
        #    ====== [End of Debugging Section] =====

        return (prediction)
    
    def line_clip(self, x1, y1, x2, y2, xwmax, ywmax):
        '''
            At first I would like to do Liang-Barsky algorithm, but it will deform the box
            So, I'll use Cohen-Sutherland Algorithm instead

                                    |                   |
                        1001        |       1000        |       1010
                                    |                   |
                 _____________(0, 0)|___________________|___________________
                                    |                   |
                                    |   (Clip window)   |
                        0001        |       0000        |       0010
                 ___________________|___________________|___________________
                                    |                   |(xwmax, ywmax)
                                    |                   |
                        0101        |       0100        |       0110
                                    |                   |
        '''
        xwmin = ywmin = 0

        if(x1 < xwmin):
            if(y1 < ywmin):
                code1 = "1001"
            elif(y1 > ywmax):
                code1 = "0101"
            else:
                code1 = "0001"
        elif(x1 > xwmax):
            if(y1 < ywmin):
                code1 = "1010"
            elif(y1 > ywmax):
                code1 = "0110"
            else:
                code1 = "0010"
        else:
            if(y1 < ywmin):
                code1 = "1000"
            elif(y1 > ywmax):
                code1 = "0100"
            else:
                code1 = "0000"
        
        if(x2 < xwmin):
            if(y2 < ywmin):
                code2 = "1001"
            elif(y2 > ywmax):
                code2 = "0101"
            else:
                code2 = "0001"
        elif(x2 > xwmax):
            if(y2 < ywmin):
                code2 = "1010"
            elif(y2 > ywmax):
                code2 = "0110"
            else:
                code2 = "0010"
        else:
            if(y2 < ywmin):
                code2 = "1000"
            elif(y2 > ywmax):
                code2 = "0100"
            else:
                code2 = "0000"

        '''
            ========= [Debugging Section] ======== [1/1]
        
        ic([xwmax, ywmax])
        ic([[x1, y1], code1])
        ic([[x2, y2], code2])
        '''
        #    ====== [End of Debugging Section] =====    

        for i in range(4):
            if (code1[i] == code2[i] == "1"): # Completely out of bounds if there is '1' in the same bit position
                return([x1, y1, x2, y2, 1])

        if (code1 == "1001" and code2 == "0110"): # Cover the entire frame
            x1 = y1 = xwmin
            x2 = xwmax
            y2 = ywmax
        elif (code1 == code2 == "0000"): # Completely Inside every boundaries
            return ([x1, y1, x2, y2, 0])
        else:
            for i in range(4): # top bottom right left
                if (code1[i] == "1"):
                    if(i == 0): y1 = 0
                    elif(i == 1): y1 = ywmax
                    elif(i == 2): x1 = xwmax
                    else: x1 = 0
                if (code2[i] == "1"):
                    if(i == 0): y2 = 0
                    elif(i == 1): y2 = ywmax
                    elif(i == 2): x2 = xwmax
                    else: x2 = 0
        return ([x1, y1, x2, y2, 2])