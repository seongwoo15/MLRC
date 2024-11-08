"""
Sample command:

python src/patch.py   \
    --train-dataset=MNIST  \
    --epochs=5  \
    --lr=0.00001  \
    --batch-size=128  \
    --model=ViT-L/14  \
    --eval-datasets=ImageNet,MNIST  \
    --results-db=results.jsonl  \
    --save=models/patch/vit_l_14  \
    --data-location=~/data \
    --alpha 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

"""
import warnings
warnings.filterwarnings('ignore')
import os, sys
# logging.info(f"Current working directory: {os.path.dirname(os.getcwd())}")
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from src.eval import evaluate
from src.finetune import finetune
from src.modeling import ImageEncoder
from src.args import parse_arguments


import logging

from datetime import datetime
from distutils.dir_util import mkpath

def patch(args):
    filename = args.results_db.replace('.jsonl', '_console.txt')  # Setting the filename from current date and time
    mkpath(os.path.dirname(filename))
    logging.basicConfig(filename=filename, filemode='a',
                        format="%(asctime)s, %(msecs)d %(name)s %(levelname)s [ %(filename)s-%(module)s-%(lineno)d ]  : %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)

    assert args.save is not None, 'Please provide a path to store models'

    # First, fine-tune    
    zeroshot_checkpoint, finetuned_checkpoint = finetune(args)

    # Load models
    zeroshot = ImageEncoder.load(zeroshot_checkpoint)
    finetuned = ImageEncoder.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        logging.info('='*100)
        logging.info(f'Evaluating with alpha={alpha:.2f}')
        args.alpha = alpha

        # interpolate between all weights in the checkpoints
        theta = {
            key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save, args.train_dataset, f'patched_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    patch(args)



