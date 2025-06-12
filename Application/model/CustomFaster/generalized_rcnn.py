"""
Implements the Generalized R-CNN framework
"""

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

from .transform import transformToImageList

import torch
from torch import nn, Tensor

from .utils import _log_api_usage_once


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be None when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        
        #bloque modificado
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-4:]
            torch._assert(
                len(val) == 4,
                f"expecting the last four dimensions of the Tensor to be num_slices, channels, H and W instead got {img.shape[-4:]}",
            )
            original_image_sizes.append((val[2], val[3]))    
        
        #bloque modificado--> no es necesaria la transformación ya que se hace en el dataloader
        #images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        #bloque modificado --> se añaden los logits del módulo de atención para que el modelo los devuelva y se tomarán como únicas imágenes los slices más representativos
        features, logits, new_images = self.backbone(images)
        new_images = transformToImageList(new_images)
        #poner comprobación de logits
        
        #bloque modificado --> ahora el parámetro images ha sido sustituido por new_images y en vez de la función images_size que no sé cual es, 
        #está la función size()
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(new_images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, new_images.image_sizes, targets)
        
        #bloque modificado--> no se hace postprocess aquí porque las imágenes ya está redimensionadas
        #detections = self.transform.postprocess(detections, new_images.size(), original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections, logits
        else:
            return self.eager_outputs(losses, detections), logits
