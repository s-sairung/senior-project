import math
import os
from matplotlib import pyplot as plt

import numpy as np
from icecream import ic

import xlsxwriter

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
    obj_class = predictions[1]
    for prediction_level, predictions in enumerate(predictions[2:]): # loop for find the same frame
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

                    obj_err = [predicted_id, obj_class, err_30, err_60, err_90]
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

def save_err_to_excel(all_warnings):
    
    workbook = xlsxwriter.Workbook('errors.xlsx')
    worksheet = workbook.add_worksheet()

    col = 0
    row = 0
    for obj_err in all_error_array:
        for level, err_level in enumerate(obj_err[2:]):
            for err_info in err_level:
                data = [obj_err[0], obj_err[1], 30+(30*level)]
                for err in err_info:
                    data.append(err)
                row += 1
                worksheet.write_row(row, col, data)

    workbook.close()
    ''''''
    workbook = xlsxwriter.Workbook('warnings.xlsx')
    worksheet = workbook.add_worksheet()

    col = 0
    row = 0
    worksheet.write_column(row, col, all_warnings)

    workbook.close()


# Plot all the errors from our prediction into the graph, for centriod errors and scale errors separately
# Also plot into frequency histograms for each level of each err type too (+ collision warning)
all_cen_err30 = []
all_cen_err60 = []
all_cen_err90 = []

all_scale_err30 = []
all_scale_err60 = []
all_scale_err90 = []
def predictPlots(all_warnings):
    
    ''' Collision Warning Histogram'''
    ic(all_warnings)
    plt.hist(np.array(all_warnings), range=[1, 3])
    plt.xlabel('Collision warning level')
    plt.ylabel('Frequency')
    plt.title('Collision Alarm Frequency')
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    eval_path = './runs/evaluation'
    eval_name = 'evaluation_warnings_demo.png'
    if not os.path.isdir(eval_path): os.makedirs(eval_path)
    plt.savefig(os.path.join(eval_path, eval_name), dpi = 300)
    plt.clf()

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
    plt.clf()

    '''Error Histogram Part1 (Centroid Error)'''
    plt.subplot(2,2,1)
    plt.hist(np.array(all_cen_err30), color = "darkblue", ec="black")
    plt.xlabel('Centroid Error')
    plt.ylabel('Frequency')
    plt.title('30 Frames ahead')
    plt.subplot(2,2,2)
    plt.hist(np.array(all_cen_err60), color = "blue", ec="darkblue")
    plt.xlabel('Centroid Error')
    plt.ylabel('Frequency')
    plt.title('60 Frames ahead')
    plt.subplot(2,2,3)
    plt.hist(np.array(all_cen_err90), color = "lightblue", ec="blue")
    plt.xlabel('Centroid Error')
    plt.ylabel('Frequency')
    plt.title('90 Frames ahead')
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    eval_path = './runs/evaluation'
    eval_name = 'evaluation_centroid_histogram_demo.png'
    if not os.path.isdir(eval_path): os.makedirs(eval_path)
    plt.savefig(os.path.join(eval_path, eval_name), dpi = 300)
    plt.clf()

    '''Error Histogram Part2 (Scale Error)'''
    plt.subplot(2,2,1)
    plt.hist(np.array(all_scale_err30), color = "red", ec="black")
    plt.xlabel('Scale Error')
    plt.ylabel('Frequency')
    plt.title('30 Frames ahead')
    plt.subplot(2,2,2)
    plt.hist(np.array(all_scale_err60), color = "orange", ec="grey")
    plt.xlabel('Scale Error')
    plt.ylabel('Frequency')
    plt.title('60 Frames ahead')
    plt.subplot(2,2,3)
    plt.hist(np.array(all_scale_err90), color = "yellow", ec="orange")
    plt.xlabel('Scale Error')
    plt.ylabel('Frequency')
    plt.title('90 Frames ahead')
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()
    eval_path = './runs/evaluation'
    eval_name = 'evaluation_scale_histogram_demo.png'
    if not os.path.isdir(eval_path): os.makedirs(eval_path)
    plt.savefig(os.path.join(eval_path, eval_name), dpi = 300)
    plt.clf()    

    '''
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

    ic(all_cen_err30)
    ic(mean_cen_30, sd_cen_30)
    ic(all_cen_err60)
    ic(mean_cen_60, sd_cen_60)
    ic(all_cen_err90)
    ic(mean_cen_90, sd_cen_90)

    ic(all_scale_err30)
    ic(mean_scale_30, sd_scale_30)
    ic(all_scale_err30)
    ic(mean_scale_60, sd_scale_60)
    ic(all_scale_err30)
    ic(mean_scale_90, sd_scale_90)
    '''
    '''
        [End of Construction Site]
    '''

