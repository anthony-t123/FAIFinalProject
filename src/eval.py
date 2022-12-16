import torch

from Dataset import Dataset
import utils
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

model = torch.load('../out/transfer0.pt')
model.eval()

dataset_test = Dataset('../Data/raw/val', get_transform(train=False))

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in data_loader_test:
        images, labels = data
        images = list(img.cuda() for img in images)
        targets = [{k: v.cuda() for k, v in t.items()} for t in labels]

        print(len(images))
        print(labels)
        # calculate outputs by running images through the network
        outputs = model(images, labels)
        print(outputs)




        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# model.eval()
#
#     running_loss = 0.0
#     loss_value = 0.0
#
#     for images, targets in dataloader:
#         images = list(img.to(device) for img in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         with torch.no_grad():
#             loss_dict = model(images)
#
#             # this returned object from the model:
#             # len is 4 (so index here), which is probably because of the size of the batch
#             # loss_dict[index]['boxes']
#             # loss_dict[index]['labels']
#             # loss_dict[index]['scores']
#             for x in range(image_batch_size):
#                 loss_value += sum(loss for loss in loss_dict[x]['scores'])
#
#         running_loss += loss_value