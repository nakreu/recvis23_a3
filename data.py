from torchvision.transforms import v2

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = {
  'train': v2.Compose([v2.ColorJitter(brightness=0.5,
                      contrast=0.5),v2.RandomHorizontalFLip(p=0.5), 
    v2.RandomVerticalFlip(p=0.2), v2.RandomPerspective(distortion_scale=0.6, p=0.5), 
    v2.RandomRotation(degrees=(0,20), v2.Resize((64, 64)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  
                   'val': v2.Compose([
    v2.Resize((64, 64)),
    v2.ToTensor(),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
}
