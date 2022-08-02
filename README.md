1: 将数据集分别放到Annotations和images文件夹中；
2：在data文件夹中修改fall.yaml文件内容，包括类别名和类别数
3：修改models文件夹下面的yolov5s.yaml文件，修改类别数
4：运行makeTxt.py文件，生成txt文件；
5: 运行voc_label.py文件，标注出训练数据
6：找到train.py文件，在train.py文件中按照下图的修改前三个路径
7：运行train.py文件
8：找到detect.py文件，修改detect.py文件的模型路径和运行的视频素材路径
9：运行detect.py文件