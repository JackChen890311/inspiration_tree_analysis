import torch
import numpy as np
import ptp.ptp_utils as ptp_utils

from typing import List
from PIL import Image
from ptp.AttentionControls import AttentionStore
from ptp.constants import *


def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    texts = []
    images_with_text = []
    for i in range(len(tokens)):
        text = decoder(int(tokens[i]))
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256), Image.NEAREST))
        images.append(image)
        texts.append(text)
        image = ptp_utils.text_under_image(image, text)
        images_with_text.append(image)
    attn_map = np.stack(images, axis=0)
    attn_map_with_text = np.stack(images_with_text, axis=0)
    ptp_utils.view_images(attn_map_with_text)
    return attn_map, texts
    

def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps.astype(np.float32) - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256), Image.NEAREST)
        image = np.array(image)
        images.append(image)
    attn_map = np.stack(images, axis=0)
    ptp_utils.view_images(attn_map)
    return attn_map