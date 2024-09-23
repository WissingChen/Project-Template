from torchvision import transforms

processer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])