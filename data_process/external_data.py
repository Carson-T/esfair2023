import pandas as pd
import numpy as np
import shutil
import random

data_path = "D:/ISIC_2019_Training_Input"
des_path = "../../external_data"
data = pd.read_csv("C:/Users/tjt/Desktop/label.csv")
des_data = pd.DataFrame()

classes = {"BCC":1000,"BKL":800,"MEL":1200,"NV":1700,"VASC":25}
for i, j in classes.items():
    new_data = data[data[i] == 1]
    class_img = random.sample(new_data["image"].values.tolist(), j)
    class_label = pd.DataFrame({"path":["../external_data/"+img+".jpg" for img in class_img],"label":i})
    des_data = des_data.append(class_label, ignore_index=True)
    for file in class_img:
        print(data_path+"/"+file)
        shutil.copy(data_path+"/"+file+".jpg", des_path+"/"+file+".jpg")

des_data.to_csv("../../external_label.csv", index=False)