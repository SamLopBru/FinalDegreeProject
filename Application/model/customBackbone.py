import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import torch
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork



class FeatureMapMerger(nn.Module):
    def __init__(self):
        super(FeatureMapMerger, self).__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=512     #es el tamaño de salida de los canales de feature map
        )

    def forward(self, logits, c3, c4, c5):

      assert logits.ndim == 2, "logits debe tener dimensiones (batch_size, num_slices)"

      fused_features_dic = {}

      list_features = [c3, c4, c5]

      # Calc weights
      weights = torch.softmax(logits, dim=1)  # (batch_size, num_slices)

      positions = torch.argmax(logits, dim=1)

      # From (batch_size, num_slices) to (batch_size, num_slices, 1, 1, 1)
      expanded_weights = weights[:, :, None, None, None]

      for i,feature_map in enumerate(list_features):

        assert feature_map.ndim == 5, "feature_maps debe tener dimensiones (batch_size, num_slices, num_feature_maps, height, width)"

        # feature_maps tiene forma (batch_size, num_slices, num_feature_maps, height, width)
        weighted_feature_map = feature_map * expanded_weights

        # Esto da como resultado un único mapa fusionado por imagen
        fused_features_dic[str(i)] = torch.sum(weighted_feature_map, dim=1)

      fpn_out = self.fpn(
          fused_features_dic
      )

      return fpn_out, positions
    
class RepresentativeSliceDetector(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, dropout=0.3):
        super(RepresentativeSliceDetector, self).__init__()
        self.feature_dim = feature_dim

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, feature_map):
        """
        Devuelve los logits de la atención y los mapas de características sin ponderar.

        Args:
            feature_map: Tensor de tamaño (batch_size, num_slices, feature_dim, H, W)

        Returns:
            feature_map: Tensor de tamaño (batch_size, num_slices, feature_dim, H, W)
            logits: Tensor de tamaño (batch_size, num_slices, 1) (logits para cada slice)
        """
        batch_size, num_slices, feature_dim, H, W = feature_map.size()

        features = feature_map.mean(dim=(3, 4))  # (batch_size, num_slices, feature_dim)

        features = self.layer_norm(features)
        logits = self.attention(features)

        return logits.squeeze(-1)
    
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT, dropout=0.3):
        super(ResNet50FeatureExtractor, self).__init__()
        self.resnet50 = models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool  # 1/4 resolución
        )

        self.layer1 = self.resnet50.layer1  # C2 (esta no la voy a usar en FPN)
        self.layer2 = self.resnet50.layer2  # C3
        self.layer3 = self.resnet50.layer3  # C4
        self.layer4 = self.resnet50.layer4  # C5

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_slices, C, H, W = x.size()
        x = x.view(batch_size * num_slices, C, H, W)

        x = self.stem(x)
        c2 = self.layer1(x)  # (B*N, 256, H/4, W/4)
        c3 = self.layer2(c2)  # (B*N, 512, H/8, W/8)
        c4 = self.layer3(c3)  # (B*N, 1024, H/16, W/16)
        c5 = self.layer4(c4)  # (B*N, 2048, H/32, W/32)

        def reshape(f): return f.view(batch_size, num_slices, f.size(1), f.size(2), f.size(3))

        return {
            'c3': reshape(c3),
            'c4': reshape(c4),
            'c5': reshape(c5)
        }

class ModifiedResNet50Backbone(nn.Module): #USA LA CAPA C4 DESPUÉS DE FPN PARA EL MRI
    def __init__(self, out_channels=2048):
        super(ModifiedResNet50Backbone, self).__init__()

        try:
          self.device = xm.xla_device()
          self.dtype = torch.bfloat16 if 'xla' in str(self.device) else torch.float32
        except:
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.dtype = torch.float16 if "cuda" in str(self.device) else torch.bfloat16


        self.out_channels = out_channels
        self.feature_extractor = ResNet50FeatureExtractor()
        self.attention_module = RepresentativeSliceDetector(feature_dim=self.out_channels)

        self.to(self.dtype)


    def forward(self, x):
        x = x.to(self.device)

        top_indices_logits = None

        features = self.feature_extractor(x)
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]

        top_indices_logits = self.attention_module(c5)

        return c3, c4, c5, top_indices_logits
    
class CustomBackbone(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = ModifiedResNet50Backbone()
    self.backbone.load_state_dict(torch.load("model/backbone.pth"))
    for param in self.backbone.parameters():
            param.requires_grad = False

    self.intermediate = FeatureMapMerger()
    self.out_channels = 512

  def forward(self, x):

    c3 ,c4 ,c5 , logits = self.backbone(x)
    fused_feature_map, positions= self.intermediate(logits, c3, c4, c5)

    representativeSlices = x[torch.arange(x.shape[0]), positions]  #son los slices más representativos


    return fused_feature_map, logits, representativeSlices
