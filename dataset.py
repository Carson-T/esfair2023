from torch.utils.data import Dataset
import os
import cv2



class MyDataset(Dataset):
    def __init__(self, path,transform):
        super(MyDataset, self).__init__()
        self.path = path
        # self.group = group

        # transform = transforms.Compose([
        #     transforms.Resize(72),  # 256
        #     transforms.CenterCrop(64),  # 224
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

        self.class_dict = {'BCC': 0, 'BKL': 1, 'MEL': 2, 'NV': 3, 'unknown': 4, 'VASC': 5} #label dictionary
        self.transform = transform
        self.data = self._make_dataset() #make dataset



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path,label = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def _make_dataset(self):
        data = []
        for class_name in self.class_dict:
            class_dir = os.path.join(self.path, class_name)
            label = self.class_dict[class_name]
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    img_path = os.path.join(class_dir, file_name)
                    data.append((img_path, label))
        return data


