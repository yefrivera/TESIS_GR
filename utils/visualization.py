# utils/visualization.py
import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx):
        # Forward
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward
        target = output[0][class_idx]
        target.backward()
        
        # Pool gradients (Global Average Pooling sobre tiempo, altura, ancho)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        
        # Ponderar activaciones
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :, :] *= pooled_gradients[i]
            
        # Promedio sobre canales y tiempo para mapa de calor 2D representativo
        heatmap = torch.mean(activations, dim=[0, 1]).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        return heatmap