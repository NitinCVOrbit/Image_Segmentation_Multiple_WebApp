import torch
import cv2
import numpy as np
import os
import torchvision.transforms as T
import segmentation_models_pytorch as smp


def create_model(class_names):
    
    # Load Pretrained U-Net with ResNet34 backbone
    model = smp.Unet(
        encoder_name="resnet34",         # Use ResNet34 as encoder
        encoder_weights="imagenet",      # Pretrained on ImageNet
        in_channels=3,                   # RGB input
        classes=len(class_names),                       
    )
        
    return model


def draw_segmentation(image, output, class_colors):
        
    # Define transparency level (0 to 1)
    alpha = 0.5  # 50% transparency

    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()

    # ✅ Fixed: Convert PIL image to NumPy before resizing
    image_np = np.array(image) 

    # ✅ Fixed: Use `.size` instead of `.shape` for PIL
    original_width, original_height = image.size

    # Resize predicted mask to match original image size
    mask = cv2.resize(predicted_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Create a colored overlay of the same shape as the image
    overlay = np.zeros_like(image_np, dtype=np.uint8)

    for class_id in range(len(class_colors)):
        if class_id == 0:
            continue  # Skip background class
        class_mask = (mask == class_id)
        color = class_colors[class_id]
        for c in range(3):
            overlay[..., c][class_mask] = color[c]
    
    # ✅ Fixed: Convert PIL image to NumPy before resizing
    image = np.array(image)  # Convert PIL to NumPy

    # Create a copy to preserve original image
    blended_image = image.copy()

    for class_id in range(len(class_colors)):
        if class_id == 0:
            continue
        class_mask = (mask == class_id)
        for c in range(3):  # For each color channel
            blended_image[..., c][class_mask] = (
                image[..., c][class_mask] * (1 - alpha) +
                overlay[..., c][class_mask] * alpha
            ).astype(np.uint8)


    return blended_image


def segmentation_unet_resnet34(image, class_names, class_colors, weights):
    
    # Load the model
    model = create_model(class_names)
    
    weights_path = os.path.join(os.path.dirname(__file__),"..","Weights",f"{weights}")
    state_dict = torch.load(weights_path, map_location = "cpu")
    model.load_state_dict(state_dict)
    
    model.eval()  # Set the model to evaluation mode
    
    # ✅ Fixed: Correct preprocessing order
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # Convert PIL to Tensor FIRST ✅
    ])
    
    # Apply model transformations
    input_tensor = transform(image).unsqueeze(0) 

    # Model Prediction
    with torch.no_grad():
        output = model(input_tensor)[0]      
    
    # Draw Predictions on the image
    image_with_segment = draw_segmentation(image, output, class_colors)

    return image_with_segment

