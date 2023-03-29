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



"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, confidences = None, names=None, colors = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]
        
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

def regression(bbox, frame):

    bbox_id = bbox[-1]

    #ic(bbox_id, regression_id)
    #ic(bbox_id not in regression_id)

    if bbox_id not in regression_id:
        regression_box = PredictionBox(bbox, frame)
        regression_dets.append(regression_box)
        regression_id.append(regression_box.id)
    else:
        regression_dets[regression_id.index(bbox_id)].update(bbox, frame)

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
    prediction =            [id, frames_ahead, (x1, y1), (x2, y2), (width, height), (xc, yc), trajectory, scaling_factor]

    predicted_results =     [prediction_k(1), prediction_k(2), prediction_k(3), ..., prediction_k(n)]

    predicted_id =          [           id_1,               id_2,                       id_3,        ...,            id_n       ]
    all_predicted_results = [[prediction_results_1], [prediction_results_2], [prediction_results_3], ..., [prediction_results_n]]
'''

def regression_prediction(frames_ahead, video_dimension):
    for regression_box in regression_dets:
        prediction = regression_box.predict_ahead(frames_ahead, video_dimension)
        # prediction: [id, frames_ahead, (width, height), (xc, yc), trajectory, delta_scale]

        if(prediction != -1):                     # -1 means the box has yet reach the minimum frames threshold
            id = prediction[0]
            if(id not in predicted_id):
                predicted_dets.append(prediction)
                predicted_id.append(id)
                all_predicted_results.append([prediction])
            else:
                predicted_dets[predicted_id.index(id)] = prediction
                predicted_results = all_predicted_results[predicted_id.index(id)]
                if(len(predicted_results) > frames_ahead):
                    predicted_results.pop(0)
                predicted_results.append(prediction)
                all_predicted_results[predicted_id.index(id)] = predicted_results
    
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

            trajectory = prediction[-3]
            scale = prediction[-2]            
            scale_change = prediction[-1]

            print("Status of the Prediction box")
            ic([id, predicted_frame, x1, y1, x2, y2, scale, scale_change])
            ''' 
            #    ====== [End of Debugging Section] =====

'''
    This fuction will take the ground truth bbox with it frame
    And a prediction of the same object for this frame
'''

'''
    obj_err         = [centroid_err, scale_err]
    frame_err_rates = [obj_err1, obj_err2, obj_err3, ..., obj_errn]
    avg_err_rate    = [avg_cen_err, avg_area_err]
    avg_err_rates   = [avg_err_rate1, avg_err_rate2, avg_err_rate1, ..., avg_err_raten]

'''
frame_err_rates = []
avg_err_rates = []
def regression_analyzer (ground_truth_bbox, frame, predictions):

    for prediction in predictions:
        predicted_frame = prediction[1]

        if(predicted_frame < frame):
            break

        if(frame == predicted_frame):
            ic(prediction)
            decimal_points = 4

            # Euclidean distance
            pred_wh = prediction[4]
            pred_area = pred_wh[0] * pred_wh[1]

            pred_centroid = prediction[5]

            gnd_x1 = ground_truth_bbox[0]
            gnd_y1 = ground_truth_bbox[1]
            gnd_x2 = ground_truth_bbox[2]
            gnd_y2 = ground_truth_bbox[3]

            gnd_xc = round((gnd_x1 + gnd_x2)/2, decimal_points)
            gnd_yc = round((gnd_y1 + gnd_y2)/2, decimal_points)

            gnd_w = round(gnd_x2 - gnd_x1, decimal_points)
            gnd_h = round(gnd_y2 - gnd_y1, decimal_points)

            gnd_area = round(gnd_w * gnd_h, decimal_points)

            centroid_err = math.dist([gnd_xc, gnd_yc], pred_centroid)
            delta_area = round(pred_area - gnd_area, decimal_points)
            area_err = delta_area/gnd_area * 100

            obj_err = [centroid_err, area_err, prediction[0]]
            frame_err_rates.append(obj_err)
            ic(frame_err_rates)
            return frame_err_rates

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

                # ic(dets_to_sort)

                if opt.track:
  
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    '''
                        tracked_dets is already predicted 1 frame ahead.
                        So what if I can feed them back to do further frame ahead?

                        tracked_dets and dets_to_sort aren't the same structure, should I be worrying?

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
                    #ic([video_width, video_height])

                    frames_ahead = 30               # [EDIT here!!] for inter/extrapolation
                    for bbox in tracked_dets:
                        regression(bbox, frame)
                    regression_prediction(frames_ahead, [video_width, video_height])
                    
                    for bbox in tracked_dets:
                        if(frame >= frames_ahead * 2):
                            id = bbox[-1]
                            if(id in predicted_id):
                                predictions = all_predicted_results[predicted_id.index(id)]
                                frame_err_rates = regression_analyzer(bbox, frame, predictions)

                    if(frame >= frames_ahead * 2):
                        sum_centroid_err = 0
                        sum_area_err = 0
                        for error in frame_err_rates:
                            sum_centroid_err += error[0]
                            sum_area_err += error[1]
                        avg_err_rates.append([sum_centroid_err/len(frame_err_rates), sum_area_err/len(frame_err_rates), frame])

                        '''
                        ========= [Debugging Section] ======== [3/3]
                        '''
                        ic(frame_err_rates)    
                        ic(avg_err_rates)
                    
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


