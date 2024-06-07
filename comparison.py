from PIL import Image
import numpy as np

def create_accuracy_image(grayscale_image_path, rgba_image_path, output_image_path):
    # Load images
    grayscale_image = Image.open(grayscale_image_path).convert('L')
    rgba_image = Image.open(rgba_image_path).convert('RGBA')
    
    # Convert images to numpy arrays
    grayscale_array = np.array(grayscale_image)
    rgba_array = np.array(rgba_image)
    
    # Ensure both images have the same dimensions
    assert grayscale_array.shape == rgba_array.shape[:2], "Images must have the same dimensions"
    
    # Create an empty image for the output
    accuracy_image = Image.new('RGB', grayscale_image.size)
    accuracy_array = np.array(accuracy_image)
    
    # Define conditions
    conditions = [
        ((0, 63), (0, 128, 0, 255)),
        ((64, 127), (255, 255, 0, 255)),
        ((128, 191), (255, 165, 0, 255)),
        ((192, 256), (255, 0, 0, 255))
    ]
    
    # Iterate through each pixel and apply conditions
    for y in range(grayscale_array.shape[0]):
        for x in range(grayscale_array.shape[1]):
            grayscale_value = grayscale_array[y, x]
            rgba_value = tuple(rgba_array[y, x])
            
            correct = False
            for (g_range, rgba_match) in conditions:
                if g_range[0] <= grayscale_value <= g_range[1] and rgba_value == rgba_match:
                    correct = True
                    break
            
            if correct:
                accuracy_array[y, x] = (0, 255, 0)  # Green for correct
            else:
                accuracy_array[y, x] = (255, 0, 0)  # Red for incorrect
    
    # Convert accuracy array back to image and save
    accuracy_image = Image.fromarray(accuracy_array)
    accuracy_image.save(output_image_path)

# Usage example
create_accuracy_image('OrnekResimler/orn3.png', 'ornek3torchOut_multiclass.png', 'output_pixel_values.png')
