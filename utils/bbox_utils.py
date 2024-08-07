def get_center_bbox(bbox):
    """
    Get the center of the bounding box
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_width_bbox(bbox):
    """
    Get the width of the bounding box
    """
    x1,y1,x2,y2 = bbox
    return x2-x1