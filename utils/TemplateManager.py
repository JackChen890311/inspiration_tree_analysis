import random

class TemplateManager:
    # https://github.com/NeuralTextualInversion/NeTI/blob/main/constants.py
    RECONTEXTUAL = [
        # "A photo of a {}",
        "A photo of {} in the jungle",
        "A photo of {} on a beach",
        "A photo of {} in Times Square",
        "A photo of {} in the moon",
        "A painting of {} in the style of Monet",
        "Oil painting of {}",
        "A Marc Chagall painting of {}",
        "A manga drawing of {}",
        'A watercolor painting of {}',
        "A statue of {}",
        "App icon of {}",
        "A sand sculpture of {}",
        "Colorful graffiti of {}",
        "A photograph of two {} on a table",
    ]

    IMAGENET_TEMPLATES_SMALL = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    IMAGENET_STYLE_TEMPLATES_SMALL = [
        "a painting in the style of {}",
        "a rendering in the style of {}",
        "a cropped painting in the style of {}",
        "the painting in the style of {}",
        "a clean painting in the style of {}",
        "a dirty painting in the style of {}",
        "a dark painting in the style of {}",
        "a picture in the style of {}",
        "a cool painting in the style of {}",
        "a close-up painting in the style of {}",
        "a bright painting in the style of {}",
        "a cropped painting in the style of {}",
        "a good painting in the style of {}",
        "a close-up painting in the style of {}",
        "a rendition in the style of {}",
        "a nice painting in the style of {}",
        "a small painting in the style of {}",
        "a weird painting in the style of {}",
        "a large painting in the style of {}",
    ]

    @classmethod
    def get_random_template(cls, temp_name):
        if temp_name not in {"RECTX", "IMG", "IMG_STY"}:
            raise ValueError(f"""Template name {temp_name} not recognized. 
                             Please use one of the following: 'RECTX', 'IMG', 'IMG_STY'""")
        if temp_name == "RECTX":
            return random.choice(cls.RECONTEXTUAL)
        if temp_name == "IMG":
            return random.choice(cls.IMAGENET_TEMPLATES_SMALL)
        if temp_name == "IMG_STY":
            return random.choice(cls.IMAGENET_STYLE_TEMPLATES_SMALL)