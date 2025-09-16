import timm
from torch import nn

def get_vit_model(num_classes: int, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    model = timm.create_model(model_name, pretrained=pretrained)
    # algunas versiones usan model.head, otras classifier
    if hasattr(model, "head"):
        in_feats = model.head.in_features
        model.head = nn.Linear(in_feats, num_classes)
    elif hasattr(model, "classifier"):
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, num_classes)
    else:
        raise RuntimeError("Modelo ViT con cabeza inesperada.")
    return model
