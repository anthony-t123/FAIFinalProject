import math
import pickle
import sys
import time

import torchvision, torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

import transforms as T
import utils
from Dataset import Dataset
from utils import reduce_dict


# For transfer learning, replacing head and retraining
def get_model_instance_transfer(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, pretrained=True, trainable_backbone_layers=1, num_classes=num_classes)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has ten classes
    num_classes = 10
    # use our dataset and defined transformations
    dataset = Dataset('Data/raw/train', get_transform(train=True))
    dataset_test = Dataset('Data/raw/val', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    transfer_model = get_model_instance_transfer(num_classes)

    # move model to the right device
    transfer_model.to(device)

    # construct an optimizer
    params = [p for p in transfer_model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.0005,
    #                             momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.0005,
                                 weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # train for one epoch, printing every 10 iterations
    losses = train_one_epoch(transfer_model, optimizer, data_loader, data_loader_test, device)
    with open("transfer_losses", "wb") as fp:
        pickle.dump(losses, fp)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

    # Save the model!
    torch.save(transfer_model, f'out/transfer.pt')

    print("That's it!")


def train_one_epoch(model, optimizer, train, validate, device):
    print(f'Starting training...')
    model.train()
    total = 0
    sum_loss = 0
    index = 0
    start_time = time.time()
    loss_list = []

    for images, targets in train:

        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())


        # target = target[0]
        # boxes = target["boxes"]
        # labels = target["labels"]
        # image_id = target["image_id"]
        # print(f'image type: {type(image)} image: {image}')

        # (image, labels, boxes) = (image.cuda(), labels.cuda(), boxes.cuda())

        # loss_dict = model(image, target)


        # losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(targets)
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if index % 100 == 0:
            curr_time = time.time()
            print(f'progress: {index}/{len(train)}', end=' ')
            loss_list.append(loss_dict_reduced)
            for label, loss in loss_dict_reduced.items():
                print(f'{label}: {loss},', end=' ')

            if index == 0:
                eta = 100000
            else:
                eta = (curr_time - start_time) * (len(train)/index)
            time_remaining = eta - (curr_time - start_time)

            print(f'    Estimated time remaining: {time.strftime("%H:%M:%S", time.gmtime(time_remaining))}')
        index += 1

    return loss_list
    # batch = y_class.shape[0]
    # x = x.cuda().float()
    # y_class = y_class.cuda()
    # y_bb = y_bb.cuda().float()
    # out_class, out_bb = model(x)
    # loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
    # loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
    # loss_bb = loss_bb.sum()
    # loss = loss_class + loss_bb/C
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # total += batch
    # sum_loss += loss.item()
    # train_loss = sum_loss/total
    #
    # return sum_loss/total


if __name__ == "__main__":
    main()
