import os
import time

import torch

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
import logging
import src.datasets as datasets


def finetune(args):
    # Check if checkpoints already exist
    zs_path = os.path.join(args.save, args.train_dataset, 'checkpoint_0.pt')
    ft_path = os.path.join(args.save, args.train_dataset, f'checkpoint_{args.epochs}.pt')
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        logging.info(f'Skipping fine-tuning because {ft_path} exists.')
        return zs_path, ft_path

    assert args.train_dataset is not None, "Please provide a training dataset."
    if args.load is not None and args.load.endswith('pt'):
        image_encoder = ImageEncoder.load(args.load)
    else:
        logging.info('Building image encoder.')
        image_encoder = ImageEncoder(args, keep_lang=False)

    classification_head = get_classification_head(args, args.train_dataset)

    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        args.train_dataset,
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    gradient_accumulation_steps = 2
    total_steps = 2000 * gradient_accumulation_steps

    devices = list(range(torch.cuda.device_count()))
    logging.info(f'Using devices {devices}')
    model = torch.nn.DataParallel(model, device_ids=devices)

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, total_steps)

    # Saving model
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        model_path = os.path.join(args.save, args.train_dataset, f'checkpoint_0.pt')
        model.module.image_encoder.save(model_path)

    step = 0

    while step < total_steps:
        model.train()
        model = model.cuda()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=None)

        for i, batch in enumerate(data_loader):
            start_time = time.time()

            if (step >= total_steps):
                break
            if (i + 1) % gradient_accumulation_steps == 0:
                scheduler(step)
                optimizer.zero_grad()
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

            batch_time = time.time() - start_time
            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                logging.info(f"Train Steps: {step} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t")
                logging.info(f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}")
            step = step + 1

        image_encoder = model.module.image_encoder

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, args.train_dataset, f'checkpoint_{step}.pt')
            image_encoder.save(model_path)
            optim_path = os.path.join(args.save, args.train_dataset, f'optim_{step}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        evaluate(image_encoder, args)

    if args.save is not None:
        zs_path = os.path.join(args.save, args.train_dataset, 'checkpoint_0.pt')
        ft_path = os.path.join(args.save, args.train_dataset, f'checkpoint_{args.epochs}.pt')
        return zs_path, ft_path

if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
