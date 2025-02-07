# Image-recognition-chatbot-
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from google.colab import files

# This will open a dialog to upload the image from your local machine
uploaded = files.upload()

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


predict_step(['sunflower.jpg'])

# Updated predict_step with attention mask
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        try:
            # Load the image from a file or URL
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return

    # Extract pixel values and attention mask
    inputs = feature_extractor(images=images, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    
    # Create attention mask
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long).to(device)

    # Generate caption with attention mask
    output_ids = model.generate(pixel_values, attention_mask=attention_mask, **gen_kwargs)

    # Decode and return predictions
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Example usage with the local image 'sunflower.jpg'
predict_step(['sunflower.jpg'])
captions = predict_step(['sunflower.jpg'])
print(captions)
