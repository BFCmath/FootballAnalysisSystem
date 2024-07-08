from ultralytics import YOLO 


model = YOLO('models/best.pt') 
#8n is much faster but 5su is more accurate

results = model.predict('videos/input.mp4',save=True)
# print(results[0])
print("-----------------------")
# for box in results[0].boxes:
#     print(box)