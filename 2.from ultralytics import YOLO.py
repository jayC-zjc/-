from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='trafic_data/data.yaml',  # 改成和你文件夹一模一样的名字！
    epochs=30,
    imgsz=640,
    batch=8
)