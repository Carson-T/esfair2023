import os
import cv2

sets = ["TrainingSet"]
groups = ["G6","G7","G8","G10"]
classes = ["BCC","BKL","MEL","NV","unknown","VASC"]
ori_path = "../../data"
des_path = "../../preprocessed_data"

for i in sets:
    pathi = ori_path+"/"+i
    path2i = des_path+"/"+i
    os.mkdir(path2i)
    for j in groups:
        pathj = pathi+"/"+j
        path2j = path2i+"/"+j
        os.mkdir(path2j)
        for k in classes:
            pathk = pathj+"/"+k
            path2k = path2j+"/"+k
            os.mkdir(path2k)
            list_dir = os.listdir(pathk)
            for img_name in list_dir:
                img_path = os.path.join(pathk,img_name)
                img = cv2.imread(img_path)
                if img.shape[0]>=3000 or img.shape[1]>=3000:
                    img = cv2.resize(img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.2)))
                elif img.shape[0]>=2000 or img.shape[1]>=2000:
                    img = cv2.resize(img,(int(img.shape[1]*0.5),int(img.shape[0]*0.5)))

                cv2.imwrite(os.path.join(path2k,img_name),img)
