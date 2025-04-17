import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline


class DiffusionUtils:
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=weight_dtype, 
        safety_checker=None, 
        requires_safety_checker=False
    )
    pipe.to(device)
    pipe.text_encoder.text_model.embeddings.token_embedding.weight.requires_grad_(False)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    new_token = {}
    orig_embeddings = pipe.text_encoder.text_model.embeddings.token_embedding.weight.clone().detach()

    @classmethod
    def run_prompt(cls, prompt, num_images_per_seed, output_path=None, seed=0):
        with torch.no_grad():
            torch.manual_seed(seed)
            images = cls.pipe(prompt=[prompt] * num_images_per_seed, num_inference_steps=25, guidance_scale=7.5).images
            np_images = np.hstack([np.asarray(img) for img in images])
            plt.figure(figsize=(10,10))
            plt.imshow(np_images)
            plt.axis("off")
            plt.title(prompt)
            plt.show()

        out_size = np_images.shape[0] # 512
        if output_path is not None:
            for i in range(num_images_per_seed):
                file_path = os.path.join(output_path, f"{prompt}_{i}.png")
                cv2.imwrite(file_path, cv2.cvtColor(np_images[:, out_size*i:out_size*(i+1), :], cv2.COLOR_RGB2BGR))

    @classmethod
    def add_new_vocab(cls, placeholder_token, embeddings):
        cls.pipe.tokenizer.add_tokens(placeholder_token)
        placeholder_token_id = cls.pipe.tokenizer.convert_tokens_to_ids(placeholder_token)

        cls.pipe.text_encoder.resize_token_embeddings(len(cls.pipe.tokenizer))
        token_embeds = cls.pipe.text_encoder.get_input_embeddings().weight.detach().requires_grad_(False)

        token_embeds[placeholder_token_id] = torch.nn.Parameter(embeddings)
        cls.new_token[placeholder_token] = placeholder_token_id
        print("New token added: ", placeholder_token)
        print("Current vocab size: ", len(cls.pipe.tokenizer))
        print("All new tokens: ", cls.new_token)


    @classmethod
    def reset_vocab(cls):
        cls.pipe.tokenizer = cls.pipe.tokenizer.__class__.from_pretrained(cls.pipe.tokenizer.name_or_path)
        cls.pipe.text_encoder.text_model.embeddings.token_embedding.weight.data = cls.orig_embeddings.clone().detach()
        cls.new_token.clear()
        
        print("Vocabulary has been reset.")
        print("Current vocab size: ", len(cls.pipe.tokenizer))


    @classmethod
    def image2latent(cls, image):
        """
        Encodes an image into latent space using the VAE encoder.
        
        Args:
            vae: The variational autoencoder (VAE) model.
            image: A numpy array representing the image, with shape (B, H, W, C) and values in [0, 255].
        
        Returns:
            latent: The encoded latent representation.
        """
        vae = cls.vae
        image = torch.tensor(image, dtype=torch.float16) / 255.0  # Normalize to [0, 1]
        image = (image - 0.5) * 2  # Scale to [-1, 1]
        image = image.permute(0, 3, 1, 2).to(vae.device)  # Convert to (B, C, H, W)
        
        with torch.no_grad():
            latent = vae.encode(image)['latent_dist'].mean  # Get latent distribution mean
        
        latent = latent * 0.18215  # Scale factor used in diffusion models
        return latent.detach().cpu().numpy()

    @classmethod
    def latent2image(cls, latent):
        """
        Decodes a latent into image space using the VAE decoder.
        
        Args:
            vae: The variational autoencoder (VAE) model.
            latent: A numpy array representing the latent, with shape (B, H//8, W//8, C).
            
        Returns:
            image: The decoded image representation.
        """
        vae = cls.vae
        latent = torch.tensor(latent).to(vae.device)
        latent = 1 / 0.18215 * latent
        image = vae.decode(latent)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image