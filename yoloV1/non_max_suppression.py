
"""
# intersection over union
"""

def IOU(left_box, right_box):
    # lt - left top coords, rb - right bottom coords
    x1_lt, y1_lt,  x1_rb, y1_rb = left_box
    x2_lt, y2_lt,  x2_rb, y2_rb = right_box
    
    # coordinates of the area of intersection.
    i_x1 = max(x1_lt, x2_lt)
    i_y1 = max(y1_rb, y2_rb)
    i_x2 = max(x1_lt, x2_lt)
    i_y2 = max(x1_rb, x2_rb)

    i_height = max(i_y2 - i_y1 + 1, 0.0)
    i_width = max(i_x2 - i_x1 + 1, 0.)
     
    area_of_intersection = i_height * i_width

    # left width, height
    height1 = y1_lt - y1_rb + 1
    width1 = x1_rb - x1_lt + 1
     
    # right width, height
    height2 = y2_lt - y2_rb + 1
    width2 = x2_rb - x2_lt + 1
     
    area_of_union = height1 * width1 + height2 * width2 - area_of_intersection

    return area_of_intersection / area_of_union



"""
Non-maximum Suppression: The final step of filtering
This is the final step of filtering bounding box candidates, so for one object, it will be only one bounding box. NMS algorithm uses B - list of bounding boxes candidates, C - list of their confidence values, and τ - the threshold for filtering overlapped bounding boxes. Let R be the result list:

1. Sort B according to their confidence scores in ascending order. 

2. Put the predicted box for B with the highest score (let it be a candidate) into R and remove it from B. 

3. Find all boxes from B with IoU(candidate, box) > τ and remove them from B. 

4. Repeat steps 1-3 until B is not empty.


"""
import torch


def nms(boxes, confidences, overlapping_threshold):
    # Sort the indexes based on confidence scores
    indexes = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    # Sort boxes according to the sorted indexes
    boxes = [boxes[i] for i in indexes]
    result = []
    while boxes:
        # Take the box with the highest confidence
        current_box = boxes.pop(0)
        result.append(current_box)
        
        # Remove boxes that have an IoU greater than the threshold
        boxes = [box for box in boxes if IOU(box, current_box) < overlapping_threshold]
    
    return result




# using torch vision
from torchvision.ops import nms

boxes = torch.tensor([[511, 41, 577, 76], [544, 59, 610, 94]], dtype=torch.float)
scores = torch.tensor([1.0, 0.5], dtype=torch.float)
overlapping_treshold = 0.5

result = nms(boxes, scores, overlapping_treshold)