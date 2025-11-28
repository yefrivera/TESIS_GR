# models/feature_extractors.py
import torch
import torch.nn as nn
from torchvision import models

class SpatialCNN(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=None, pretrained=True):
        """
        Wrapper para backbones de extracción de características espaciales.
        Args:
            model_name: 'resnet18', 'mobilenet_v3_large', 'efficientnet_b0'
            num_classes: Si no es None, añade una capa FC para clasificación estática.
            pretrained: Usar pesos de ImageNet (Transfer Learning).
        """
        super(SpatialCNN, self).__init__()
        
        if model_name == 'resnet18':
            # ResNet-18: Buen balance, usado en benchmarks de HaGRID [cite: 2552]
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 512
            # Remover la capa FC original para usarlo como extractor
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        elif model_name == 'mobilenet_v3':
            # MobileNetV3: Optimizado para CPU/Edge devices [cite: 2552]
            self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = 960
            # Remover clasificador
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        else:
            raise ValueError(f"Modelo {model_name} no soportado aún.")

        # Clasificador opcional para Gestos Estáticos (HaGRID)
        self.classifier = None
        if num_classes:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        features = self.backbone(x)
        
        if self.classifier:
            return self.classifier(features)
        
        return torch.flatten(features, 1) # Retorna vector de características