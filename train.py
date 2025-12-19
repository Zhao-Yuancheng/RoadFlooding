from ultralytics import YOLO

# 这里有三种训练方式，三种任选其一

# 第一种：根据yaml文件构建一个新模型进行训练,若对YOLO8网络进行了修改（比如添加了注意力机制）适合选用此种训练方式。但请注意这种训练方式是重头训练(一切参数都要自己训练),训练时间、资源消耗都是十分巨大的
# model = YOLO('yolov8n.yaml')  # build a new model from YAML

# 第二种：加载一个预训练模型，在此基础之前上对参数进行调整。这种方式是深度学习界最最主流的方式。由于大部分参数已经训练好，我们仅需根据数据集对模型的部分参数进行微调，因此训练时间最短，计算资源消耗最小。
model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

# 第三种:根据yaml文件构建一个新模型，然后将预训练模型的参数转移到新模型中，然后进行训练，对YOLO8网络进行改进的适合选用此种训练方式，而且训练时间不至于过长
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# data参数指定数据集yaml文件(我这里data.yaml与train、val文件夹同目录)
# epochs指定训练多少轮
# imgsz指定图片大小
# results = model.train(data='wheat.yaml', epochs=100, imgsz=640, batch=16, workers=0, device=0)  # train the model
results = model.train(data='wheat.yaml', epochs=100, imgsz=640, batch=16, workers=0)