import torchvision
import torch

from functools import partial
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from constants import BATCH_SIZE

def create_model(num_classes=BATCH_SIZE):
    r"""_summary_ Create model based off RetinaNet50 model with customizations

    Args:
        num_classes (int, optional): _description_. input batch size. Defaults to {BATCH_SIZE}.

    Returns:
        model: Customized RetinaNet50 model
    """
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    )
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model

if __name__ == '__main__':
    model = create_model(BATCH_SIZE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum( p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")