import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def load_and_preprocess_lr_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

with open('saved_model/generator60.json', 'r') as json_file:
    generator_json = json_file.read()

with open('saved_model/discriminator60.json', 'r') as json_file:
    discriminator_json = json_file.read()

generator = model_from_json(generator_json)
discriminator = model_from_json(discriminator_json)

generator.load_weights('saved_model/generator60_weight.hdf5')
discriminator.load_weights('saved_model/discriminator60_weight.hdf5')

lr_image = load_and_preprocess_lr_image('rsz_dog1.jpg', (56, 56))  # Implement this function
sr_image = generator.predict(np.expand_dims(lr_image, axis=0))
sr_image = sr_image.squeeze()

# Expand dimensions only if discriminator model expects batch dimension
if len(discriminator.input_shape) == 4:
    sr_image = np.expand_dims(sr_image, axis=0)


# Now predict using the discriminator
discriminator_prediction = discriminator.predict(sr_image)
# print(sr_image)

sr_image_int = (sr_image * 255).astype('uint8')

gen_img = Image.fromarray(sr_image_int[0])
gen_img.save("gen_img.png")