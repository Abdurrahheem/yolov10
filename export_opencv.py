from ultralytics import YOLOv10

print("LOADING MODEL")
model = YOLOv10.from_pretrained("jameslahm/yolov10s")

print("EXPORTING MODEL TO ONNX")
model.export(format='onnx', simplify=False, imgsz=(480, 640))

print("FORWARD PASS")
results = model.predict(source="./sample-image.png",  conf=0.5)
print("END")
