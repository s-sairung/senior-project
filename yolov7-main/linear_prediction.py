from icecream import ic

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

        scale:          Area of bbox
        trajectories:   Is an queue that store the bbox centriod (world coordinates from real picture) for the last 30 frame prior
                        (I use queue becuase we can dequeue the oldest one out then put newest in for updated prediction)
        times_tracked:  Count the times that this object get tracked 
        


    '''

    def __init__(self, bbox):
        
        worldXcentroid = (bbox[0] + bbox[2])//2
        worldYcentroid = (bbox[1] + bbox[3])//2
        centroidarr = [worldXcentroid, worldYcentroid]

        self.x = abs(bbox[0] - bbox[2])
        self.y = abs(bbox[1] - bbox[3])
        self.scales = [30]
        self.scales.append(self.x * self.y)
        self.trajectories = [30]
        self.trajectories.append(centroidarr)
        self.times_tracked = 1

    def predict_ahead(self):
        predict_bbox = self.predict_ahead(self)

    def update(self, new_bbox):
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
            self.trajectories.pop(0)
            self.trajectories.append(newCentroidarr)
