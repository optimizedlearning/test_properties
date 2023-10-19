# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import wandb
import datasets
import evaluate
import torch
import time
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from PIL import Image
from tqdm.auto import tqdm
from datasets import disable_caching
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to datasets",
        required=False,
    )
    ## Path to the datasets that we store locally
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to locally stored datasets",
        required=False,
    )
    ## Set streaming option (we might need to stream large dataset)
    parser.add_argument(
        "--streaming",
        type=bool,
        default=False,
        help="Set streaming option",
        required=False,
    )
    

    ## specify starting epoch if necessary
    parser.add_argument(
        "--starting_epoch",
        type=int,
        default=0,
        help="Current training epoch",
    )
    ## specify training step if necessary
    parser.add_argument(
        "--starting_step",
        type=int,
        default=0,
        help="Current training step",
    )
    ####
    parser.add_argument(
        "--name",
         type=str,
        default="very_cool_experiment"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=90, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--resume_from_checkpoint_torch",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument("--output_epoch_dir", type=str, default=None, help="Where to store the model at the end of epoch")

    args = parser.parse_args()
    
    # Sanity checks
    if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def compute_linear_approx(model, prev_param):
    linear_approx = 0
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            # print("p.grad", torch.norm(p.grad))
            # print("norm difference",torch.norm(p.add(-prev_param[i])) )
            linear_approx+= torch.dot(torch.flatten(p.grad), torch.flatten(p.add(-prev_param[i]))).item()
            i+=1
    return linear_approx

def compute_smoothness(model, prev_param, prev_grad):
    sum_num = 0
    sum_denom = 0
    i=0
    with torch.no_grad():
        for p in model.parameters():
            sum_num += torch.norm(p.grad - prev_grad[i])
            sum_denom += torch.norm(p - prev_param[i])
            i+=1
    return sum_num/sum_denom
    
class CustomImageDataset(datasets.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]["file_path"])
        return {"image": image, "label": self.data[idx]["label"]}

def main():
    args = parse_args()
    if args.data_dir is not None: 
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(args.data_dir)
    wandb.init(project='clean_propeties_run', config=args, name=args.name)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_image_classification_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        disable_caching()
        dataset = load_dataset(args.dataset_name, task="image-classification" ,streaming = True,  use_auth_token=True)
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
            task="image-classification",
        )
    # disable_caching()
    # dataset_train = load_dataset( 'imagenet-1k',task = 'image-classification', streaming = True, use_auth_token=True)
    # dataset_train = dataset_train.with_format("torch")
    # dataset_val = load_dataset( '/projectnb/aclab/tranhp/Imagenet/val/', streaming = True)
    # dataset_val = dataset_val.w
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    # args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    # if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
    #     split = dataset["train"].train_test_split(args.train_val_split)
    #     dataset["train"] = split["train"]
    #     dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["label"].names
    # labels = dataset_train.features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and image processor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        i2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
        trust_remote_code=args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    # model = AutoModelForImageClassification.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    #     trust_remote_code=args.trust_remote_code,
    # )
    model = AutoModelForImageClassification.from_config(config)
    wandb.watch(model)
    # Preprocessing the datasets

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
            # dataset_train = dataset_train.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset["train"].with_transform(preprocess_train)
        train_dataset = dataset["train"].map(preprocess_train, batched= True)
        # train_dataset = dataset_train.with_transform(preprocess_train)
        if args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
            # dataset_val = dataset_val.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        # Set the validation transforms
        # eval_dataset = dataset["validation"].with_transform(preprocess_val)
        eval_dataset = dataset["validation"].map(preprocess_val, batched= True)
        # eval_dataset = dataset_val.with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Get the metric function
    metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(enumerate(train_dataloader))
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # print("path", path)
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        # training_difference = os.path.splitext(path)[0]

        # if "epoch" in training_difference:
        #     starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        #     resume_step = None
        #     completed_steps = starting_epoch * num_update_steps_per_epoch
        # else:
        #     # need to multiply `gradient_accumulation_steps` to reflect real steps
        #     resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        #     starting_epoch = resume_step // len(train_dataloader)
        #     completed_steps = resume_step // args.gradient_accumulation_steps
        #     resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    prev_grad = [torch.zeros_like(p) for p in model.parameters()]
    prev_param = [torch.zeros_like(p) for p in model.parameters()] 
    current_grad = [torch.zeros_like(p) for p in model.parameters()]
    current_param = [torch.zeros_like(p) for p in model.parameters()] 
    resume_step = args.starting_step
    for epoch in range(args.starting_epoch, args.num_train_epochs):
        model.train()
        convexity_gap = 0
        L = 0
        num = 0
        denom = 0
        prev_loss = 0
        current_loss = 0
        iteration= 0
        total_loss = 0
        exp_avg_L_1 = 0
        exp_avg_L_2 = 0
        exp_avg_gap_1 = 0
        exp_avg_gap_2 = 0
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        if args.resume_from_checkpoint_torch is not None:
            checkpoint = torch.load(args.resume_from_checkpoint_torch)
            convexity_gap = checkpoint['convexity_gap']
            L = checkpoint['smoothness']
            num = checkpoint['num']
            denom = checkpoint['denom']
            prev_loss = checkpoint['prev_loss']
            current_loss = checkpoint['current_loss']
            iteration= checkpoint['iteration']
            total_loss = checkpoint['total_loss']
            exp_avg_L_1 = checkpoint['exp_avg_L_.99']
            exp_avg_L_2 = checkpoint['exp_avg_L_.9999']
            exp_avg_gap_1 = checkpoint['exp_avg_gap_.99']
            exp_avg_gap_2 = checkpoint['exp_avg_gap_.9999']
        start = time.time()
        for step, batch in enumerate(active_dataloader):
            iteration +=1
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if step >0:
                    prev_loss = current_loss
                    # print("prev s", current_loss)
                    # print("current loss", loss.detach().float() )
                    i = 0
                    with torch.no_grad():
                        for p in model.parameters():
                            prev_grad[i].copy_(current_grad[i])
                            prev_param[i].copy_(current_param[i])
                            i+=1
                    # get the inner product
                    linear_approx = compute_linear_approx(model, current_param)
                    # get the smoothness constant, small L means function is relatively smooth
                    current_L = compute_smoothness(model, current_param, current_grad)
                    L = max(L,current_L)
                    # this is another quantity that we want to check: linear_approx / loss_gap. The ratio is positive is good
                    num+= linear_approx
                    denom+= loss.detach().float() - current_loss
                    current_convexity_gap = loss.detach().float() - current_loss - linear_approx 
                    exp_avg_gap_1 = 0.99*exp_avg_gap_1 + (1-0.99)*current_convexity_gap
                    exp_avg_gap_2 = 0.9999*exp_avg_gap_2 + (1-0.9999)*current_convexity_gap
                    exp_avg_L_1 = 0.99*exp_avg_L_1+ (1-0.99)*current_L
                    exp_avg_L_2 = 0.9999*exp_avg_L_2+ (1-0.9999)*current_L
                    # print("denom", denom)
                    # compute convexity gap, negative means convex
                    convexity_gap+= current_convexity_gap
                i = 0
                with torch.no_grad():
                    for p in model.parameters():
                        current_grad[i].copy_(p.grad)
                        current_param[i].copy_(p)
                        i+=1
                optimizer.step()
                current_loss =  loss.detach().float()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (step+1) % 10000 == 0 or (step ==1 and resume_step == 0):
                    dic = {'prev_loss': prev_loss, 'prev_grad': prev_grad, 'prev_param': prev_param, 'current_loss': current_loss, 'current_grad': current_grad
                           , 'current_param': current_param, 'smoothness': L, 'num': num , 'denom': denom, 'convexity_gap': convexity_gap, 'iteration': iteration,
                           'exp_avg_L_.99': exp_avg_L_1,'exp_avg_L_.9999': exp_avg_L_2,  "exp_avg_gap_.99":  exp_avg_gap_1, "exp_avg_gap_.9999":  exp_avg_gap_2,
                           'total_loss': total_loss}

                    print("saving babyyy!")
                    # output_dir_epoch = f"epoch_{epoch}"
                    # output_dir_step = f"epoch_{step}"
                    if args.output_dir is not None:
                        output_dir = args.output_dir + str(epoch) + "_" + str(step+resume_step)
                    #### Changed!
                    ## Save the dictionary
                    store_name = args.output_dir + str(epoch)+ "_" +str(step +resume_step) + ".pth.tar"
                    torch.save(dic, store_name)
                    ####
                    accelerator.save_state(output_dir)
                    end = time.time()
                    wandb.log(
                        {
                            "train_loss": total_loss.item() / iteration,
                            "step": completed_steps,
                            "convexity_gap": convexity_gap/iteration,
                            "smoothness": L,
                            "linear/loss_gap": num/denom,
                            "learning_rate": lr_scheduler.get_lr()[0],
                            "numerator" : num,
                            "denominator": denom,
                            "samples_per_sec": 100*args.per_device_train_batch_size/(end-start),
                            'exp_avg_L_.99': exp_avg_L_1,
                            'exp_avg_L_.9999': exp_avg_L_2, 
                            "exp_avg_gap_.99":  exp_avg_gap_1, 
                            "exp_avg_gap_.9999":  exp_avg_gap_2
                        }
                    )
                ###
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            repo.push_to_hub(
                                commit_message=f"Training in progress {completed_steps} steps",
                                blocking=False,
                                auto_lfs_prune=True,
                            )
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(eval_metric)
        logger.info(f"epoch {epoch}: {eval_metric}")
        #### Changed!
        ## We also want to log data at the end of any epoch
        wandb.log(
                {
                    "accuracy": eval_metric['accuracy'],
                    "train_loss_end": total_loss.item() / iteration,
                    "epoch": epoch,
                }
            )
        ##
        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / iteration,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )
        ## The checkpoint dictionary of the original github doesn't have the loss. This is an extra dictionary to store the loss.
        dic = {'prev_loss': prev_loss, 'prev_grad': prev_grad, 'prev_param': prev_param, 'current_loss': current_loss, 'current_grad': current_grad
                           , 'current_param': current_param, 'smoothness': L, 'num': num , 'denom': denom, 'convexity_gap': convexity_gap, 'iteration': iteration,
                           'exp_avg_L_.99': exp_avg_L_1,'exp_avg_L_.9999': exp_avg_L_2,  "exp_avg_gap_.99":  exp_avg_gap_1, "exp_avg_gap_.9999":  exp_avg_gap_2,
                           'total_loss': total_loss}
        ####
        if args.checkpointing_steps == "epoch":
            print("saving babyyy!")
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_epoch_dir, output_dir)
            #### Changed!
            ## Save the dictionary
            store_name =args.output_epoch_dir + str(epoch)+".pth.tar"
            torch.save(dic, store_name)
            ####
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)


if __name__ == "__main__":
    main()
