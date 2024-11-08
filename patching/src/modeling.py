import torch

import clip.clip as clip

from src import utils
import logging

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        logging.info(f'Loading {args.model} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logging.info(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        logging.info(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logging.info(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        logging.info(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logging.info(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        logging.info(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
