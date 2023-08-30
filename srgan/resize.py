from PIL import Image
import os

# Path to the folder containing the images
input_folder = 'original_good_images'

# Path to the folder where resized images will be saved
output_folder = 'img_align_celeba'

# Desired dimensions for resized images
new_width = 224
new_height = 224

# Iterate through each image file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        
        # Resize the image using the default filter (nearest)
        resized_img = img.resize((new_width, new_height))
        
        # Save the resized image to the output folder
        output_path = os.path.join(output_folder, filename)
        resized_img.save(output_path)

print("Resizing complete.")
