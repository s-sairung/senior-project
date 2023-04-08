import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *
from linear_prediction import *
import math
import matplotlib.pyplot as plt



"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat] if colors is not None else 0
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ":"+ names[cat] if identities is not None else  f'{names[cat]} {confidences[i]:.2f}'
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    return img
'''
def draw_prediction_boxes(img, bbox, identities=None, warning = False):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        if(warning):
            color = (0,0,211)
        else:
            color = (211,211,211)
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img
'''

def draw_prediction_boxes(img, all_predictions):
    #ic(all_predictions)
    id = int(all_predictions[0])
    for prediction_level, predictions in enumerate(all_predictions[1:]):
        if(len(predictions) < 1): 
            break

        box = predictions[0]
        x1 = int(box[1][0])
        y1 = int(box[1][1])
        x2 = int(box[2][0])
        y2 = int(box[2][1])
        warning = box[5]

        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        ttc = ""
        if(warning):
            color = (0,0,211- prediction_level*30)
            if(prediction_level == 0):
                ttc = " [Warning] Collision within 1 second!!!"
            elif(prediction_level == 1):
                ttc = " [Warning] Collision within 2 seconds!!"
            elif(prediction_level == 2):
                ttc = " [Warning] Collision within 3 seconds!"
        else:
            color = (211- prediction_level*30 ,211- prediction_level*30 ,211- prediction_level*30)
        
        if not opt.nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        if not opt.nolabel:
            label = str(id) + ttc
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

'''
    [Construction Site Ahead] [1/2]

    regression_dets:    array that stores the bbox objects from class PredictionBox
    regerssion_id:      array that stores the bbox_id (bbox_label)

        [!!!WARNING!!!] This area is still in development, expected the worst as of now!!!

    regression_id =     [       id_1,           id_2,               id_3,       ...,        id_n     ]
    regression_dets =   [regression_box_1, regression_box_2, regression_box_3,  ..., regression_box_n]

    regression_box: look at PredictionBox's object initialization for more details
'''
regression_dets = []
regression_id = []
countdown = [] # time since last updated frame

def regression(bbox, frame):

    bbox_id = bbox[-1]

    if bbox_id not in regression_id:
        regression_box = PredictionBox(bbox, frame)
        regression_dets.append(regression_box)
        regression_id.append(regression_box.id)
        countdown.append(0)
    else:
        regression_dets[regression_id.index(bbox_id)].update(bbox, frame)
        countdown[regression_id.index(bbox_id)] = 0 

def regression_checker (limit):
    times_deleted = 0    
    for index in range(len(countdown)):
        countdown[index] += 1
        if countdown[index] > limit:
            regression_dets.pop(index)
            regression_id.pop(index)
            countdown.pop(index)
            countdown.append(-1)
            times_deleted += 1
            index -= 1
    while times_deleted > 0:
        countdown.pop(-1)
        times_deleted -= 1

    '''
        ========= [Debugging Section] ======== [1/2]
                        
    print("Status of the Regression box")
    ic([regression_box.id, regression_box.times_tracked, regression_box.x, regression_box.y])
    ic(regression_box.frames)
    ic(regression_box.trajectories)
    ic(regression_box.scales)
    '''
    #    ====== [End of Debugging Section] =====
    
predicted_dets = []
all_predicted_results = []
predicted_id = []

'''
    predicted_id =          [           id_1,               id_2,                 id_3,        ...,          id_n       ]
    all_predicted_results = [prediction_results_1, prediction_results_2, prediction_results_3, ..., prediction_results_n]

    predicted_results =     [id, prediction_30, prediction_60, prediction_90]
    predicted_k =           [prediction_k(1), prediction_k(2), prediction_k(3), ..., prediction_k(n)]
    prediction =            [predicted_frame, (x1, y1), (x2, y2), (width, height), (xc, yc), collision, trajectory, scaling_factor]
'''

def regression_prediction(frames_ahead, video, video_dimension):
    for regression_box in regression_dets:

        predicted_results = regression_box.predict_ahead(frames_ahead, video_dimension)
        if(predicted_results != -1):        # -1 means the box has yet reach the minimum frames threshold
            id = predicted_results[0]
            if(id not in predicted_id):
                predicted_id.append(id)
                all_predicted_results.append(predicted_results)
            else:
                prediction_30 = predicted_results[1][0]
                prediction_60 = []
                prediction_90 = []

                if(len(predicted_results[2]) > 0):
                    prediction_60 = predicted_results[2][0]
                    if(len(predicted_results[3]) > 0):
                        prediction_90 = predicted_results[3][0]

                if(len(prediction_30) > 0):
                    all_predicted_results[predicted_id.index(id)][1].append(prediction_30)
                    if(len(prediction_60) > 0):
                        all_predicted_results[predicted_id.index(id)][2].append(prediction_60)
                        if(len(prediction_90) > 0):
                            all_predicted_results[predicted_id.index(id)][3].append(prediction_90)

            draw_prediction_boxes(video, predicted_results)
    
    '''        
                ========= [Debugging Section] ======== [2/3]
                       
            predicted_frame = prediction[1]

            x1y1 = prediction[2]
            x1 = x1y1[0]
            y1 = x1y1[1]

            x2y2 = prediction[3]
            x2 = x2y2[0]
            y2 = x2y2[1]

            width_height = prediction[4]
            width = width_height[0]
            height = width_height[1]

            centroid = prediction[5]
            xc = centroid[0]
            yc = centroid[1]

            collision = prediction[-3]
            trajectory = prediction[-2]          
            scale_change = prediction[-1]

            print("Status of the Prediction box")
            ic([id, predicted_frame, x1, y1, x2, y2, scale_change])
            ''' 
            #    ====== [End of Debugging Section] =====           

'''
    all_error_id    = [ obj_id_1, obj_id_2,  obj_id_3,  ...,  obj_id_n] <<<<< global permanent array
                            ..         ..       ..                ..
    all_error_array = [obj_err_1, obj_err_2, obj_err_3, ..., obj_err_n] <<<<< global permanent array

    obj_err_k   = [label, class, err_30, err_60, err_90]            <<<<  local reuseable array //get and update

    err_a       = [err_a_1, err_a_2, err_a_3, ..., err_a_m]     <<<<  local reuseable array //get and update

    err_a_frame     = [cen_err30, area_err, frame]                        <<<<  local reuseable array //create new everytime
'''
all_error_id = []
all_error_array = []
def regression_analyzer (ground_truth_bbox, frame, predictions):
    predicted_id = predictions[0]
    for prediction_level, predictions in enumerate(predictions[1:]): # loop for find the same frame
        #ic(predictions)
        for prediction in predictions:
            if(len(prediction) < 1):
                break
            predicted_frame = prediction[0]
            
            if(predicted_frame > frame):
                break
                
            if(frame == predicted_frame): 
                decimal_points = 4

                # Euclidean distance
                pred_wh = prediction[3]
                pred_area = pred_wh[0] * pred_wh[1]

                pred_centroid = prediction[4]

                gnd_x1 = ground_truth_bbox[0]
                gnd_y1 = ground_truth_bbox[1]
                gnd_x2 = ground_truth_bbox[2]
                gnd_y2 = ground_truth_bbox[3]

                gnd_class = ground_truth_bbox[4]

                gnd_xc = round((gnd_x1 + gnd_x2)/2, decimal_points)
                gnd_yc = round((gnd_y1 + gnd_y2)/2, decimal_points)

                gnd_w = round(gnd_x2 - gnd_x1, decimal_points)
                gnd_h = round(gnd_y2 - gnd_y1, decimal_points)

                gnd_area = round(gnd_w * gnd_h, decimal_points)

                centroid_err = math.dist([gnd_xc, gnd_yc], pred_centroid)

                delta_area = round(abs(pred_area - gnd_area), decimal_points)
                area_err = delta_area/gnd_area * 100

                err = [centroid_err, area_err, frame]

                err_30 = []
                err_60 = []
                err_90 = []
                if(predicted_id not in all_error_id):
                    if(prediction_level == 0):
                        err_30 = [err]
                    elif(prediction_level == 1):
                        err_60 = [err]
                    else:
                        err_90 = [err]

                    obj_err = [predicted_id, gnd_class, err_30, err_60, err_90]
                    all_error_id.append(predicted_id)
                    all_error_array.append(obj_err)
                else:
                    obj_err = all_error_array[all_error_id.index(predicted_id)]
                    if(prediction_level == 0):
                        err_30 = obj_err[2]
                        err_30.append(err)
                    elif(prediction_level == 1):
                        err_60 = obj_err[3]
                        err_60.append(err)
                    else:
                        err_90 = obj_err[4]
                        err_90.append(err)

#centroid_limit = 0

all_cen_err30 = []
all_cen_err60 = []
all_cen_err90 = []

all_scale_err30 = []
all_scale_err60 = []
all_scale_err90 = []
# Plot all the errors from our prediction into the graph, for centriod errors and scale errors separately
def predictPlots(video_dimension):
    
    #scale_limit = round(video_dimension[0]*video_dimension[1], 4)

    for object in all_error_array:
        object_id = object[0]
        object_class = object[1]
        object_30 = object[2]
        object_60 = object[3]
        object_90 = object[4]

        cen_err30 = []
        sca_err30 = []
        frames_plot30 = []
        for err in object_30:
            cen_err30.append(err[0])
            sca_err30.append(err[1])
            frames_plot30.append(err[2])
            all_cen_err30.append(err[0])
            all_scale_err30.append(err[1])

        cen_err60 = []
        sca_err60 = []
        frames_plot60 = []
        for err in object_60:
            cen_err60.append(err[0])
            sca_err60.append(err[1])
            frames_plot60.append(err[2])
            all_cen_err60.append(err[0])
            all_scale_err60.append(err[1])

        cen_err90 = []
        sca_err90 = []
        frames_plot90 = []
        for err in object_90:
            cen_err90.append(err[0])
            sca_err90.append(err[1])
            frames_plot90.append(err[2])
            all_cen_err90.append(err[0])
            all_scale_err90.append(err[1])

        # Centroid Error Subplot
        plt.subplot(1,2,1)
        if(len(frames_plot30) > 10):
            plt.plot(frames_plot30, cen_err30, linestyle = '-', label = 'id: ' + str(object_id) + ' class: ' + str(object_class) + ' (30)')
        if(len(frames_plot60) > 10):
            plt.plot(frames_plot60, cen_err60, linestyle = '--', label = 'id: ' + str(object_id) + ' class: ' + str(object_class) + ' (60)')
        if(len(frames_plot90) > 10):    
            plt.plot(frames_plot90, cen_err90, linestyle = ':', label = 'id: ' + str(object_id) + ' class: ' + str(object_class) + ' (90)')
        plt.xlabel('frame')
        plt.ylabel('centroid error (px)')
        plt.title('Centroid Error Graph')

        # Scale Error Subplot
        plt.subplot(1,2,2)
        if(len(frames_plot30) > 10):
            plt.plot(frames_plot30, sca_err30, label = 'id: ' + str(object_id) + ' class: ' + str(object_class) + ' (30)')
        if(len(frames_plot60) > 10):
            plt.plot(frames_plot60, sca_err60, linestyle = '--', label = 'id: ' + str(object_id) + ' class: ' + str(object_class) + ' (60)')
        if(len(frames_plot90) > 10):    
            plt.plot(frames_plot90, sca_err90, linestyle = ':', label = 'id: ' + str(object_id) + ' class: ' + str(object_class)  + ' (90)')
        plt.xlabel('frame')
        plt.ylabel('scale error (px*px)')
        plt.title('Scale Error Graph')

    #plt.subplot(1,2,1)
    #plt.gca().set_ylim([0, centroid_limit])
    #plt.subplot(1,2,2)
    #plt.gca().set_ylim([0, scale_limit])
    plt.gcf().set_size_inches(10, 5)
    #plt.legend(loc = 'upper center', bbox_to_anchor = (0., 0.), ncol = 5)
    plt.tight_layout()
    eval_path = './runs/evaluation'
    eval_name = 'evaluation_demo.png'
    if not os.path.isdir(eval_path): os.makedirs(eval_path)
    plt.savefig(os.path.join(eval_path, eval_name), dpi = 300)
    #plt.show()

    mean_cen_30 = np.mean(np.array(all_cen_err30))
    mean_cen_60 = np.mean(np.array(all_cen_err60))
    mean_cen_90 = np.mean(np.array(all_cen_err90))

    sd_cen_30 = np.std(np.array(all_cen_err30))
    sd_cen_60 = np.std(np.array(all_cen_err60))
    sd_cen_90 = np.std(np.array(all_cen_err90))

    mean_scale_30 = np.mean(np.array(all_scale_err30))
    mean_scale_60 = np.mean(np.array(all_scale_err60))
    mean_scale_90 = np.mean(np.array(all_scale_err90))

    sd_scale_30 = np.std(np.array(all_scale_err30))
    sd_scale_60 = np.std(np.array(all_scale_err60))
    sd_scale_90 = np.std(np.array(all_scale_err90))

    #ic(all_cen_err30)
    #ic(mean_cen_30, sd_cen_30)
    #ic(all_cen_err60)
    #ic(mean_cen_60, sd_cen_60)
    #ic(all_cen_err90)
    #ic(mean_cen_90, sd_cen_90)

    #ic(all_scale_err30)
    #ic(mean_scale_30, sd_scale_30)
    #ic(all_scale_err30)
    #ic(mean_scale_60, sd_scale_60)
    #ic(all_scale_err30)
    #ic(mean_scale_90, sd_scale_90)

    '''
        [End of Construction Site]
    '''
    
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:  
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    ###################################
    startTime = 0
    ###################################
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))

                if opt.track:
  
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    '''
                        tracked_dets is already predicted 1 frame ahead.

                        dets_to_sort = [[x11, y11, x12, y12, conf1, class1],
                                        [x21, y21, x22, y22, conf2, class2],
                                            .   .   .   .   .   .   .   .
                                            .   .   .   .   .   .   .   .
                                        [xk1, yk1, xk2, yk2, confk, classk]]
                            *sort by conf, high -> low*

                        whereas,                
                        tracked_dets = [[xm1, ym1, xm2, ym2, classm, um, vm, sm, labelm],
                                        [x(m-1)1, y(m-1)1, x(m-1)2, y(m-1)2, class(m-1), u(m-1), v(m-1), s(m-1), label(m-1)],
                                            .   .   .   .   .   .   .   .   .   .   .
                                            .   .   .   .   .   .   .   .   .   .   .
                                        [x11, y11, x12, y12, class1, u1, v1, s1, label1]]
                            *sort by label, high -> low*
                    '''

                    '''
                        [Construction Site Ahead] [2/2]
                    '''
                    video_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    loss_limit = 10                 # [EDIT here!!] limit for successive times didn't got detected
                    frames_ahead = 30               # [EDIT here!!] for inter/extrapolation
                    for bbox in tracked_dets:
                        regression(bbox, frame)                                      #this will create 'regression_dets'
                    regression_checker(loss_limit)
                    
                    regression_prediction(frames_ahead, im0, [video_width, video_height]) #this will create 'all_predicted_results'
                    
                    if(frame >= frames_ahead * 2):
                        for bbox in tracked_dets:   #Check the accuracy for each and every object detected in this frame
                            id = bbox[-1]
                            if(id in predicted_id):
                                predictions = all_predicted_results[predicted_id.index(id)]
                                regression_analyzer(bbox, frame, predictions)       #this will create 'all_error_array'
                    '''
                        ========= [Debugging Section] ======== [3/3]
                        
                        ic(frame_err_rates)
                        ic(sum_centroid_err)
                        ic(sum_area_err)
                        ic(len(frame_err_rates))    
                        ic(avg_err_rates)
                        '''
                        #    ====== [End of Debugging Section] =====
                    '''
                        [End of Construction Site]
                    '''
                    
                    tracks =sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            #loop over tracks
                            for t, track in enumerate(tracks):
                  
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])), 
                                                (int(track.centroidarr[i+1][0]),
                                                int(track.centroidarr[i+1][1])),
                                                track_color, thickness=opt.thickness) 
                                                for i,_ in  enumerate(track.centroidarr) 
                                                    if i < len(track.centroidarr)-1 ] 
                else:
                    bbox_xyxy = dets_to_sort[:,:4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]
                
                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

                
                    
                            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            ######################################################
            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1/(currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)

            #######################################################
            if view_img:
                cv2.namedWindow(str(p), cv2.WINDOW_KEEPRATIO) # NOTE: only works with the qt backend
                cv2.imshow(str(p), im0)
                cv2.resizeWindow(str(p), 600, 600)
                cv2.waitKey(int(not opt.pause_frame)) # if pause_frame: 0 (forever) else: 1 (1 ms)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    predictPlots([video_width, video_height])

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true', help='run tracking')
    parser.add_argument('--show-track', action='store_true', help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    parser.add_argument('--pause-frame', action='store_true', help='pause each frame')

    opt = parser.parse_args()
    print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2) 

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


