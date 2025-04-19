import numpy as np
import cv2

def gray2color(gray_image, use_pallet='default', custom_pallet=None):
    """
    Convert a grayscale image (single channel, uint8 or int type) to a color image
    using a predefined or custom color palette.
    
    Args:
        gray_image (numpy.ndarray): Grayscale/single-channel image
        use_pallet (str): Which palette to use ('default', 'cityscape', etc.)
        custom_pallet (numpy.ndarray): Custom color palette if use_pallet is not 'default'
        
    Returns:
        numpy.ndarray: Colored version of the input image
    """
    # Ensure gray_image is a numpy array
    gray_image = np.asarray(gray_image)
    
    # If image has more than 1 channel, return as is
    if len(gray_image.shape) > 2 and gray_image.shape[2] > 1:
        return gray_image
    
    # Make sure image is 2D
    if len(gray_image.shape) > 2:
        gray_image = gray_image[:, :, 0]
    
    # Default color palette (rainbow-like)
    if use_pallet == 'default':
        # Create a rainbow-like palette with 256 colors
        pallet = np.zeros((1, 256, 3), dtype=np.uint8)
        
        # Generate colors for the palette
        for i in range(256):
            if i <= 42:  # Red to yellow
                pallet[0, i] = [255, 6*i, 0]
            elif i <= 85:  # Yellow to green
                pallet[0, i] = [255-(i-43)*6, 255, 0]
            elif i <= 128:  # Green to cyan
                pallet[0, i] = [0, 255, (i-86)*6]
            elif i <= 171:  # Cyan to blue
                pallet[0, i] = [0, 255-(i-129)*6, 255]
            elif i <= 214:  # Blue to magenta
                pallet[0, i] = [(i-172)*6, 0, 255]
            else:  # Magenta to red
                pallet[0, i] = [255, 0, 255-(i-215)*6]
    else:
        # Use the provided custom palette
        if custom_pallet is not None:
            pallet = custom_pallet
        else:
            raise ValueError("If use_pallet is not 'default', custom_pallet must be provided.")
    
    # Ensure gray_image is within valid range
    gray_image = np.clip(gray_image, 0, pallet.shape[1]-1)
    
    # Convert gray to color using the palette
    # First, create an empty color image
    color_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    
    # Use the gray values as indices into the palette
    if len(pallet.shape) == 3:  # 3D palette
        # Apply the palette mapping
        for i in range(pallet.shape[1]):
            mask = gray_image == i
            if np.any(mask):
                color_image[mask] = pallet[0, i]
    
    return color_image 