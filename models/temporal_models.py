import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    """Módulo de Atención de Canal adaptado para tensores 5D (B, C, D, H, W) [cite: 982]"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # MLP compartido (usando Convoluciones 1x1x1 como sugiere el paper para reducir parámetros)
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    """Módulo de Atención Espacial con kernel 3D extendido [cite: 1021]"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # Convolución 3D para capturar dinámica espacial a través del tiempo
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compresión de canal: max y avg a través de los canales
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class DepthAttention(nn.Module):
    """
    Innovación del paper: Atención de Profundidad (Temporal)[cite: 1031].
    Focaliza en los frames más relevantes de la secuencia.
    """
    def __init__(self, depth, kernel_size=7):
        super(DepthAttention, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1) # Reducción
        # Convoluciones para capturar features temporales
        padding = kernel_size // 2
        self.conv_d1 = nn.Conv3d(1, 1, kernel_size=(depth, 1, 1), padding=(padding, 0, 0)) 
        # Nota: Ajustar padding según la profundidad exacta o usar stride para mantener dimensiones
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Promedio global espacial para enfocarse en el eje D (tiempo/profundidad)
        # El paper sugiere reducir canales primero
        avg_spatial = torch.mean(x, dim=1, keepdim=True) # (B, 1, D, H, W)
        # Simplificación para implementación: Atención sobre el eje D
        out = self.conv1x1(avg_spatial)
        return self.sigmoid(out)

class CBAM3D_Block(nn.Module):
    """Integración de los 3 submódulos [cite: 977]"""
    def __init__(self, in_planes, depth_dim, ratio=16):
        super(CBAM3D_Block, self).__init__()
        self.ca = ChannelAttention3D(in_planes, ratio)
        self.sa = SpatialAttention3D()
        # self.da = DepthAttention(depth_dim) # Opcional: activar según costo computacional

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        # out = out * self.da(out) # Aplicar atención de profundidad
        return out

class ResNeXt3D_Bottleneck(nn.Module):
    """Bottleneck estándar de ResNeXt modificado para 3D + CBAM [cite: 895]"""
    expansion = 4
    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXt3D_Bottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        
        # Convolución agrupada 3D (Grouped Convolution)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Integración del módulo 3D-CBAM propuesto en la tesis
        self.cbam = CBAM3D_Block(planes * self.expansion, depth_dim=32) # Depth debe ser dinámico
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # Aplicar atención antes del residual
        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNeXt101_3D_CBAM(nn.Module):
    def __init__(self, num_classes, sample_duration, cardinality=32):
        super(ResNeXt101_3D_CBAM, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        
        # Capa inicial (Stem)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Capas ResNeXt: Configuración [3, 4, 23, 3]
        self.layer1 = self._make_layer(ResNeXt3D_Bottleneck, 128, 3)
        self.layer2 = self._make_layer(ResNeXt3D_Bottleneck, 256, 4, stride=2)
        self.layer3 = self._make_layer(ResNeXt3D_Bottleneck, 512, 23, stride=2)
        self.layer4 = self._make_layer(ResNeXt3D_Bottleneck, 1024, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(1024 * 4, num_classes) # Expansion es 4

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x