import torch.nn as nn
from torchvision import models

def create_model(model_name: str,
                fine_tune: bool,
                num_classes: int = 196):

    model = None

    # DenseNet 201
    if model_name == 'densenet_201': # sudah
        weights = models.DenseNet201_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.densenet201(weights=weights)
        model.name = 'densenet_201'
        model.classifier = nn.Linear(in_features=model.classifier.in_features, 
                                    out_features=num_classes)
    # EFF B4
    elif model_name == 'efficientnet_b4': # sudah
        weights = models.EfficientNet_B4_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.efficientnet_b4(weights=weights)
        model.name = 'efficientnet_b4'
        model.classifier[1] = nn.Linear(in_features=1792, 
                                        out_features=num_classes)

    # EfficientNet V2 S
    elif model_name == 'efficientnet_v2_s': # sudah
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.efficientnet_v2_s(weights=weights)
        model.name = 'efficientnet_v2_s'
        model.classifier[1] = nn.Linear(in_features=1280, 
                                        out_features=num_classes)
    
    # ResNet 50
    elif model_name == 'resnet_50': # sudah
        weights = models.ResNet50_Weights.DEFAULT
        model_transform = weights.transforms()
        model = models.resnet50(weights=weights)
        model.name = 'resnet_50'
        
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)
                
            return model, model_transform

        model.fc = nn.Linear(in_features=model.fc.in_features, 
                            out_features=num_classes)
        
    else:
        raise NotImplementedError

    if fine_tune:
        print('[INFO]: Unfreezing all layers...')
        for params in model.parameters():
            params.requires_grad = True

    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.features.parameters():
            params.requires_grad = False

    return model, model_transform