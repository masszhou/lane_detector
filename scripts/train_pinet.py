from comet_ml import Experiment
import fire
from tqdm import tqdm
import time
import torch

from model.net import PINet
from dataset.builder import build_dataset
from dataset.builder import build_dataloader

from instances.trainer import TrainerLaneDetector
from instances.builder import build_validate_fn

from configs.PINet.network import NetworkParameters
from configs.PINet.training import TrainingParameters


def get_parameters(d_name: str):
    if d_name == "tusimple":
        from configs.PINet import ParamsTuSimple
        return ParamsTuSimple()
    elif d_name == "bdd100k":
        from configs.PINet import ParamsBDD100K
        return ParamsBDD100K()
    elif d_name == "culane":
        from configs.PINet import ParamsCuLane
        return ParamsCuLane()
    elif d_name == "collections":
        from configs.PINet import ParamsCollections
        return ParamsCollections()
    else:
        return None


def training(dataset: str, log_comet_ml=False):
    print('Train PINet')

    # ------------------------------------------------------------
    # 1. get parameters
    network_params = NetworkParameters()
    training_params = TrainingParameters()

    # ------------------------------------------------------------
    # 2. build dataset -> output sample dict
    dataset_params = get_parameters(dataset)
    train_dataset, val_dataset = build_dataset(dataset_params)
    validate_fn = build_validate_fn(dataset_params)

    # ------------------------------------------------------------
    # 3. build dataloader -> output batch tensor
    train_generator = None
    if train_dataset is not None:
        train_generator = build_dataloader(train_dataset, training_params.batch_size)

    # ------------------------------------------------------------
    # 4. define network
    print('Get model, train from sketch')
    model = PINet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # start from gpu_0
    model.to(device)

    # ------------------------------------------------------------
    # 5. associate network model to a training instance
    # ToDo: resume training
    lane_detector = TrainerLaneDetector(model, network_params, training_params)
    # lane_detector.load_weights(p.model_path)
    lane_detector.training_mode()

    # ------------------------------------------------------------
    # 6. define logging enviroment
    if log_comet_ml:
        print('Logging with comet_ml')
        experiment = Experiment(api_key="7XBXyqF4ctwO6HnnHnBGuv7U3",
                                project_name="lanedetection",
                                workspace="masszhou")
        experiment.log_parameters(vars(network_params))
        experiment.log_parameters(vars(training_params))
        experiment.log_parameters(vars(dataset_params))
        experiment.log_dataset_info(name=dataset)
    else:
        experiment = None

    # ------------------------------------------------------------
    # 7. start training phase
    step = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for epoch in range(training_params.num_epochs):
        pbar = tqdm(total=len(train_dataset)/training_params.batch_size)
        loss_p = -1.0
        for batch in train_generator:
            # inputs -> ndarray[#batch, 3, 256, 512]
            # target_lanes -> List[ndarray] e.g. [[4, 48],..., ]
            # target_h -> List[ndarray] e.g. [[4, 48],..., ]
            # test_image -> ndarray [3, 256, 512]
            loss_p, metrics, outputs = lane_detector.train(batch, epoch=epoch, step=step)

            # parse confidence map from result
            outputs_last_block = outputs[-1]
            confidance, _, _ = outputs_last_block  # [8, 1, 32, 64]
            confidance = confidance[0].cpu().data.numpy()
            confidance = confidance.transpose((1, 2, 0))

            if log_comet_ml:
                experiment.log_metric("train total loss", loss_p)
                experiment.log_metric("learning rate", lane_detector.get_lr())
                experiment.log_metrics(metrics)
                if step % 500 == 0:
                    # log confidence output from first image in batch
                    experiment.log_image(confidance, name="image id:{}".format(batch["image_id"][0]))

            pbar.set_description(f'epoch {epoch}')
            pbar.set_postfix(total_loss=loss_p)
            pbar.update()
            step += 1
        pbar.close()

        # save model per epoch
        file_name = f"./tmp/{timestr}_epoch-{epoch}_totalstep-{step}_loss-{loss_p:.2f}.pth"
        lane_detector.save_model_v2(file_name)

        # if epoch % 10 == 0:
        if val_dataset is not None:
            file_name = f"./tmp/{timestr}_epoch-{epoch}_validation.json"
            scores = validate_fn(dataset=val_dataset, net=lane_detector, validate_file_name=file_name, logger=experiment)


if __name__ == '__main__':
    fire.Fire(training)