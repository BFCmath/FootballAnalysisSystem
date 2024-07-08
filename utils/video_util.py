import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done reading video")
            break
        frames.append(frame)
        
    return frames

def save_video(frames, output_path):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")