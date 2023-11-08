import fiftyone as fo
import fiftyone.zoo as foz

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torch.nn.functional import pad

from PIL import Image

# Step 1 : Load the dataset
# download the following dataset in /Users/sorenmarcelino/fiftyone/open-images-v7/train
dataset = foz.load_zoo_dataset(
    "coco-2017", split="validation",
    max_samples=100,
)
print(dataset)
session = fo.launch_app(dataset.view())  # visualize dataset in fiftyone tool


# Step 2 : Define a custom dataset class
class CustomCocoDataset(Dataset):
    def __init__(self, fiftyone_dataset, transform=None):
        self.fiftyone_dataset = fiftyone_dataset
        self.transform = transform
        self.ids = fiftyone_dataset.values("id")
        # create a mapping for labels to integer IDs if they are strings
        self.label_to_int = {label: idx for idx, label in
                             enumerate(fiftyone_dataset.distinct("ground_truth.detections.label"))}

    def __len__(self):
        return len(self.fiftyone_dataset)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample = self.fiftyone_dataset[sample_id]  # get the fiftyone sample
        image = Image.open(sample.filepath).convert("RGB")  # load the image
        detections = sample.ground_truth.detections  # get the detections
        # convert the detections to the target format
        boxes = []
        labels = []
        area = []
        iscrowd = []
        for detection in detections:
            # get the bounding box in [xmin, ymin, xmax, ymax] format
            bbox = detection.bounding_box
            xmin = bbox[0] * image.width
            ymin = bbox[1] * image.height
            xmax = (bbox[0] + bbox[2]) * image.width
            ymax = (bbox[1] + bbox[3]) * image.height
            boxes.append([xmin, ymin, xmax, ymax])
            # convert the string label to an integer ID
            label_int = self.label_to_int[detection.label]
            labels.append(label_int)  # get the label

            area.append((xmax - xmin) * (ymax - ymin))  # calculate the area
            iscrowd.append(detection.iscrowd)  # determine if the instance is a crowd

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # apply the transformations
        if self.transform:
            image = self.transform(image)

        # prepare the target dictionnary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


# Step 3 : Define your transformations
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize all images to the same size
    transforms.ToTensor(),
])


def custom_collate_fn(batch):
    # pad images to have the same size and create batches
    batch_images = []
    batch_targets = []
    for image, target in batch:
        batch_images.append(image)

        # update the target dictionnary to tensor, no need to stack
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        batch_targets.append(target)

    return torch.stack(batch_images, 0), batch_targets


# Step 4 : Create the Dataset and DataLoader
coco_dataset = CustomCocoDataset(dataset, transform=transform)
data_loader = DataLoader(coco_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

# Step 5 : Define the model, loss function and optimizer
model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)  # load a pre-trained Faster R-CNN model

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 91  # COCO dataset has 80 classes + 1 background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.boxpredictor = detection_models.fasterrcnn_resnet50_fpn(
    pretrained=True).roi_heads.box_predictor.__class__(in_features, num_classes)

# define loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()  # this not may be needed as loss is handled inside the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# move model to the appropriate device (GPU if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Step 6 : Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        # move the batch of images and targets to the device used
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward pass
        loss_dict = model(images, targets)

        # the loss is a sum of all of the losses for all of the outputs
        losses = sum(loss for loss in loss_dict.values())

        # backward pass and optimize
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch}/{num_epochs}, Loss: {losses}')

# Step 7 : Validate the model
# implement the validation loop
model.eval()

# Save the model
torch.save(model.state_dict(), 'model_weights.pth')
