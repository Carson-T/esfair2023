import torch
import albumentations
from albumentations import pytorch as AT
import cv2
from torchvision import models
from model import mymodel



def predict(model,img):
    model.eval()
    output = model(img)
    _, pred = torch.max(output, dim=1)
    return pred

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_path = r"C:\Users\tjt\Desktop\projects\esfair\sample_code-main\data\ValSet\G6\BCC\0060643.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = albumentations.Compose([
        albumentations.Resize(72, 72),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    img = transform(image=img)["image"]


    img = img.to(device)
    img = torch.unsqueeze(img, 0)
    pretrained_model = models.resnet50(pretrained=True)
    model = mymodel(pretrained_model, 6)

    model.load_state_dict(torch.load('./saved_model/resnet50-fp16-v3.pth',map_location=device))
    model.to(device)

    output = predict(model,img)

    # torch.onnx.export(model, img, "./onnx/resnet50-v1.onnx")