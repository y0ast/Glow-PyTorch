import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from datasets import get_CIFAR10, get_SVHN
from mydataset import get_CelebA_data, CelebALoader, get_test_conditions, get_new_test_conditions, CLEVRDataset
from model import Glow
from evaluator import evaluation_model
from utils import save_image
import wandb


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses



def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}
    y_logits = torch.sigmoid(y_logits)
    y = y.float()
    loss_classes = F.binary_cross_entropy_with_logits(y_logits, y)
    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def main(
    dataset,
    dataroot,
    download,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
    classifier_weight
):

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    wandb.init(project=args.dataset)

    check_manual_seed(seed)

    image_shape = (64,64,3)
    if args.dataset == "task1": num_classes = 24
    else : num_classes = 40

    # Note: unsupported for now
    multi_class = True #It's True but this variable doesn't be used now


    if args.dataset == "task1":
        dataset_train = CLEVRDataset(root_folder=args.dataroot,img_folder=args.dataroot+'images/')
        train_loader = DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,drop_last=True)
    else :
        dataset_train = CelebALoader(root_folder=args.dataroot) #'/home/yellow/deep-learning-and-practice/hw7/dataset/task_2/'
        train_loader = DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,drop_last=True)



    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )

    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    wandb.watch(model)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)
        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            ### x: torch.Size([batchsize, 3, 64, 64]); y: torch.Size([batchsize, 24]); z: torch.Size([batchsize, 48, 8, 8])
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        wandb.log({"loss": losses["total_loss"].item()})

        return losses


    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=None, require_empty=False
    )
    ### n_saved (Optional[int]) – Number of objects that should be kept on disk. Older files will be removed. If set to None, all objects are kept.

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )


    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    ## load pre-trained model if given
    # if saved_model:
    #     model.load_state_dict(torch.load(saved_model))
    #     model.set_actnorm_init()
    #
    #     if saved_optimizer:
    #         optimizer.load_state_dict(torch.load(saved_optimizer))
    #
    #     file_name, ext = os.path.splitext(saved_model)
    #     resume_epoch = int(file_name.split("_")[-1])
    #
    #     @trainer.on(Events.STARTED)
    #     def resume_training(engine):
    #         engine.state.epoch = resume_epoch
    #         engine.state.iteration = resume_epoch * len(engine.state.dataloader)
    if saved_model:
        model.load_state_dict(torch.load(saved_model, map_location="cpu")['model'])
        model.set_actnorm_init()

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)




    evaluator = evaluation_model(args.classifier_weight)
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        if args.dataset == "task1":
            model.eval()
            with torch.no_grad():
                test_conditions = get_test_conditions(args.dataroot).cuda()
                tmp_x = torch.rand( ( len(test_conditions) , image_shape[2], image_shape[0], image_shape[0]) ).cuda()
                z, _, _ = model(tmp_x, test_conditions)
                z = torch.randn(z.size()).cuda()
                predict_x = model(y_onehot=test_conditions, z=z, temperature=1, reverse=True)
                score = evaluator.eval(predict_x, test_conditions)
                save_image(predict_x, args.output_dir+f"/Epoch{engine.state.epoch}_score{score:.3f}.png")

                new_test_conditions = get_new_test_conditions(args.dataroot).cuda()
                new_predict_x = model(y_onehot=new_test_conditions, z=z, temperature=1, reverse=True)
                new_score = evaluator.eval(new_predict_x, new_test_conditions)
                save_image(predict_x, args.output_dir+f"/Epoch{engine.state.epoch}_newscore{new_score:.3f}.png")

                losses = ", ".join([f"{key}: {value:.2f}" for key, value in engine.state.metrics.items()])
                print(f"Iter: {engine.state.iteration}  score:{score:.3f} newscore:{new_score:.3f} {losses}")
                wandb.log({"score": score, "new_score": new_score})


    # @trainer.on(Events.EPOCH_COMPLETED)
    # def evaluate(engine):
    #     evaluator.run(test_loader)
    #
    #     scheduler.step()
    #     metrics = evaluator.state.metrics
    #
    #     losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
    #
    #     print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")
    #
    # timer = Timer(average=True)
    # timer.attach(
    #     trainer,
    #     start=Events.EPOCH_STARTED,
    #     resume=Events.ITERATION_STARTED,
    #     pause=Events.ITERATION_COMPLETED,
    #     step=Events.ITERATION_COMPLETED,
    # )

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def print_times(engine):
    #     pbar.log_message(
    #         f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
    #     )
    #     timer.reset()

    trainer.run(train_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="task2",
        choices=["task1", "task2"],
        help="Type of the dataset to be used.",
    )

    parser.add_argument("--dataroot", type=str, default="./", help="path to dataset")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=6, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size used during training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="number of epochs to train for"
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    parser.add_argument(
        "--classifier_weight",
        default="",
        help="full path of classifier_weight for task1",
    )

    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (
            len(os.listdir(args.output_dir)) > 0
        ):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    main(**kwargs)
