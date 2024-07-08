from ultralytics import YOLO
import supervision as sv
import cv2
import pickle 
import os
from utils import get_width_bbox, get_center_bbox
import numpy as np
class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self,frames,batch_size=20):
        # predict object in all frames
        detected_frames = []
        for i in range(0,len(frames),batch_size):
            batch = frames[i:i+batch_size] # get a batch of frames
            results = self.model.predict(batch,conf=0.1) 
            detected_frames.extend(results)
        print("Done detecting frames")
        return detected_frames
        
    def get_object_tracks(self, frames, read_from_stub=False,stub_path=None):
        print("Start tracking")
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("Stub does exist, reading from stub...")
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detected_frames = self.detect_frames(frames)
        
        # tracks (dict)
        ## "players"/ "referees"/ "ball" (list)
        ### frame_id (dict) 
        #### track_id (dict) 
        ##### "boxx": [x1,y1,x2,y2]
        
        # tracks["players"][frame_id][track_id]["boxx"] = [x1,y1,x2,y2]
        tracks = {
            "players":[],
            "ball":[],
            "referees":[],
        }
        for frame_id, detection in enumerate(detected_frames):
            cls_names = detection.names # map (id:name)
            cls_names_inv = {v:k for k,v in cls_names.items()} # create a map (name:id)
            
            #convert to sv format
            detection_sv  = sv.Detections.from_ultralytics(detection)
            
            #convert goalkeeper to player
            for object_id, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[object_id] = cls_names_inv["player"]
                    
            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv) #nearly the same with SORT

            # for each frame, append the tracks
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
            
            # the detections are in the format (number detected)*[box(xyxy), ..., ..., class_id, trace_id]
            for detect in detection_with_tracks:
                boxx = detect[0].tolist()
                cls_id = detect[3]
                trace_id = detect[4]
                
                if cls_names[cls_id] == "player":
                    tracks["players"][frame_id][trace_id] = {"boxx":boxx}
                if cls_names[cls_id] == "referee":
                    tracks["referees"][frame_id][trace_id] = {"boxx":boxx}
                    
            for detect in detection_sv:
                boxx = detect[0].tolist()
                cls_id = detect[3]
                trace_id = detect[4]
                if cls_names[cls_id] == "ball":
                    tracks["ball"][frame_id][1] = {"boxx":boxx} #only one ball
                    
        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(tracks,f)
            print("Stub saved at", stub_path)    
        print("Done tracking")
        return tracks
    
    def draw_triangle(self, frame, bbox, color):
        x_center, _ = get_center_bbox(bbox)
        y1 = int(bbox[1])
        
        # Define the triangle points
        triangle = np.array([
            [int(x_center), int(y1)],
            [int(x_center - 10), int(y1 - 20)],
            [int(x_center + 10), int(y1 - 20)]
        ])
        
        # Draw the filled triangle
        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        
        # Draw the triangle border
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        
        return frame
    
    def draw_ellipse(self,frame,bbox,color,id=None):
        x_center,_ = get_center_bbox(bbox)
        width = get_width_bbox(bbox)
        y2 = int(bbox[3])
        
        #draw ellipse
        cv2.ellipse(img=frame, center=(int(x_center),int(y2)), axes=(int(width), int(width*0.35)), angle=0, startAngle=-45, endAngle=235, color=color, thickness=1, lineType=cv2.LINE_4)
        
        #draw id
        rec_width = 30
        rec_height = 15
        x1_rect = int(x_center - rec_width//2)
        x2_rect = int(x_center + rec_width//2)
        y1_rect = int((y2-rec_height//2)+15) # +15 to move the rectangle down
        y2_rect = int((y2+rec_height//2)+15) # +15 to move the rectangle down
        
        if id is not None:
            cv2.rectangle(frame, (x1_rect,y1_rect), (x2_rect,y2_rect), color, cv2.FILLED)
            x1_text = x1_rect+12
            y1_rect = y1_rect+15
            if id > 99:
                x1_text-10  
            cv2.putText(frame, str(id), (x1_rect,y1_rect), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
        return frame
    def draw_annotations(self,frames,tracks):
        print("Drawing annotations")
        output_video_frames = [] # list of frames with annotations
        for frame_id, frame in enumerate(frames): #draw on each frame
            frame = frame.copy() # make a copy of the frame
            player_dict = tracks["players"][frame_id]
            ball_dict = tracks["ball"][frame_id]
            referee_dict = tracks["referees"][frame_id]
            
            # Draw elip for players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame,player["boxx"],player["color"],track_id)
            
            # Draw elip for referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["boxx"],(0,0,255))
            
            # Draw triangle for ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["boxx"],(255,0,0))
            
            output_video_frames.append(frame)
        return output_video_frames