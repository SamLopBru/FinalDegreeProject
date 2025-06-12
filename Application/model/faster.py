import torch.nn as nn
import torch
import torchvision
from .CustomFaster.anchor_utils import AnchorGenerator
from .CustomFaster.faster_rcnn import FasterRCNN as CustomFasterRCNN
from .customBackbone import CustomBackbone


class CustomFaster(nn.Module):
  """
  CustomFaster is a modification of FasterRCNN, with a custom backbone and a intermediate module.
  Is made to be trained in a TPU and can distinguish between background and anomaly. When inference,
  the model will select as an anomaly does outputs with more that a 0.5 score and those which boxes is
  composed of less than 95% of dark pixels.
  """

  def __init__(self):

    super().__init__()
    self.backbone = CustomBackbone()
    try:
      self.device = xm.xla_device()
      self.dtype = torch.bfloat16 if 'xla' in str(self.device) else torch.float32

    except:
      if torch.cuda.is_available():
        self.device = torch.device("cuda")
        self.dtype = torch.float16
      else:
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    anchor_generator = AnchorGenerator(
      sizes = ((8, 16, 256), (32, 48, 256), (64, 96, 256)),  #el 256 es por los casos actionable
      aspect_ratios=((0.5, 1.0, 1.5),(0.5, 1.0, 1.5),(0.5, 1.0, 1.5))
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
      featmap_names=['0', '1', '2'], output_size=5, sampling_ratio=2
    )

    self.fasterRCNN = CustomFasterRCNN(
        backbone=self.backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
      )
    
  

  def forward(self, slice_tensor, targets=None):
    """
    slices_tensor: Slices tensor of size (batch_size, num_slices, C, H, W).
    targets: Dic with 'boxes' and 'labels' (only used when training).
    """

    slice_tensor = slice_tensor.to(self.dtype)
    
    if self.training:
      if targets is None or not isinstance(targets, list) or not all(isinstance(t, dict) for t in targets):
        raise ValueError(f"`Targets` should be a dic list, but {type(targets)} was passed")
      targets = [{k: v.to(self.dtype) if torch.is_tensor(v) and k != "labels"
                           else v for k, v in t.items()} for t in targets]
      losses, logits =  self.fasterRCNN(slice_tensor,targets )
      return losses, logits

    else:

      outputs, logits = self.fasterRCNN(slice_tensor)

      threshold_black_ratio = 0.90
      pixel_threshold = 0.1
      outputs = outputs[0]
      scores = outputs['scores']
      
      if scores.size(0) == 0:
        return outputs, logits
      
      else:
        max_indice = torch.argmax(scores)
        label = outputs['labels'][max_indice]
        box = outputs['boxes'][max_indice].round().int()
        x1, y1, x2, y2 = box.tolist()
        scores = scores[max_indice]
        cropped_region = slice_tensor[0, :, :,y1:y2, x1:x2]

        black_pixel_ratio = (cropped_region < pixel_threshold).float().mean()

        if black_pixel_ratio >= threshold_black_ratio:
          return {
              'boxes': torch.zeros((0, 4), device=self.device),
              'scores': torch.tensor([1.0], device=self.device),
              'labels': torch.tensor([0], device=self.device)
          }, logits

        else:
          return {
              'boxes': box,
              'scores': scores,
              'labels': label
          }, logits
        
  
      


