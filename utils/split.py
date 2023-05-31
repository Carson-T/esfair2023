from sklearn import model_selection
import os
import shutil
groups = ["G6","G7","G8","G10"]
classes = ["BCC","BKL","MEL","NV","unknown","VASC"]
path = "../data/TrainingSet"
test_path = "../data/ValSet"
for i in range(4):
    path_per_group = os.path.join(path,groups[i])
    test_path_per_group = os.path.join(test_path,groups[i])
    for j in range(6):
        path_per_class = os.path.join(path_per_group,classes[j])
        test_path_per_class = os.path.join(test_path_per_group,classes[j])
        # os.mkdir(test_path_per_class)
        files = os.listdir(path_per_class)
        train_files,test_files = model_selection.train_test_split(files,train_size=0.8)
        for file in test_files:
            img_path = os.path.join(path_per_class,file)
            shutil.move(img_path,test_path_per_class)
