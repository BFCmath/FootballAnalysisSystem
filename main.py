from utils import *
from tracker import *
from team_assigner import *
def main():
    # read video
    frames = read_video("videos/input.mp4")
    # track objects
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(frames,read_from_stub=True,stub_path="stubs/tracks.pkl")
    
    # team assigner
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],tracks["players"][0]) #only get players of the first frame 
    
    for frame_id, player_detections in enumerate(tracks["players"]):
        for player_id, player_detection in player_detections.items():
            player_bbox = player_detection["boxx"]
            player_team = team_assigner.get_player_team(frames[frame_id],player_bbox,player_id)
            #assign the team to player_detection
            player_detection["team"] = player_team
            #assign the color to player_detection
            player_detection["color"] = team_assigner.team_colors[player_team]
    
    # draw annotations
    output_frames = tracker.draw_annotations(frames,tracks)
    
    # save video
    save_video(output_frames, "videos/output1.mp4")
    pass

if __name__ == "__main__":
    main()
