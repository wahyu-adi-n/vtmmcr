import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from comet_ml import Artifact, Experiment
from utils.utils import generate_model_config, read_cfg, get_optimizer, \
      get_device, generate_hyperparameters, save_model, save_plots, SaveBestModel, \
      set_seeds, print_train_time
from data.dataset import VehicleMakeModelDataset, VehicleColorDataset
from tqdm.auto import tqdm
from models.models import create_model
from utils.logger import get_logger


def train_one_epoch(model: torch.nn.Module, 
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, 
                    criterion: torch.nn.Module,
                    device: torch.device,
                    epoch: int):
    print('[1] Training process...')

    train_loss, train_acc = 0.0, 0.0

    model.train()

    for batch, (X, y) in tqdm(enumerate(train_loader), 
                        total=len(train_loader)):

        image, labels = X.to(device), y.to(device)

        outputs = model(image)

        loss = criterion(outputs, labels)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        train_acc += (y_pred_labels == labels).sum().item()/len(outputs)

    train_loss /=  len(train_loader)
    train_acc /= len(train_loader)

    # Log metric for train loss and accuracy
    logger.log_metric("train_loss", train_loss, epoch=epoch)
    logger.log_metric("train_acc", train_acc, epoch=epoch)

    return train_loss, train_acc


def validate_one_epoch(model: torch.nn.Module, 
                      val_loader: torch.utils.data.DataLoader, 
                      criterion: torch.nn.Module, 
                      device: torch.device,
                      epoch: int):
    
    print('[2] Validation process...')

    val_loss, val_acc = 0.0, 0.0
    model.eval()

    with torch.no_grad():

        for batch, (X, y) in tqdm(enumerate(val_loader), 
                                  total=len(val_loader)):

            image, labels = X.to(device), y.to(device)

            outputs = model(image)
                        
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_pred_labels = outputs.argmax(dim=1)
            val_acc += ((val_pred_labels == labels).sum().item()/len(val_pred_labels))

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    
    # Log metric for val loss and accuracy
    logger.log_metric("val_loss", val_loss, epoch=epoch)
    logger.log_metric("val_accuracy", val_acc, epoch=epoch)
    
    return val_loss, val_acc

def train(model: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader,  
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          device: torch.device, 
          cfg):

    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }

    epochs = cfg['train']['num_epochs'] 
    
    model.to(device)

    for epoch in range(1, epochs+1):
        print(f"[INFO]: Epoch {epoch} of {epochs}")
        train_loss, train_acc = train_one_epoch(model,
                                                train_loader, 
                                                optimizer, 
                                                criterion, 
                                                device, 
                                                epoch)
        val_loss, val_acc = validate_one_epoch(model, 
                                              val_loader, 
                                              criterion, 
                                              device, 
                                              epoch)

        print(
          f"Epoch: {epoch} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}"
        )

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)


        save_best_model(model, val_loss, epoch, cfg)

    save_model(model, cfg)

    save_plots(results['train_loss'], 
                results['train_acc'], 
                results['val_loss'], 
                results['val_acc'], cfg)
    
    print('TRAINING COMPLETE')

    return results, model 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', 
                        default="../Trainer/configs/effnetb1.yaml",
                        type=str, help="Path to config yaml file")
                        
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    hyperparameters = generate_hyperparameters(cfg)

    LOG = get_logger(cfg['model']['backbone'])

    LOG.info("Training Process Start")
    logger = Experiment(api_key=cfg['logger']['api_key'],
                        project_name=cfg['logger']['project_name'],
                        workspace=cfg['logger']['workspace']) 

    artifact = Artifact("VTMMCR Artifact", "Model")
    LOG.info("Comet Logger has successfully loaded.")

    device = get_device(cfg)
    LOG.info(f"{str(device)} has choosen.")

    print(f"\nComputation device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Epochs: {cfg['train']['num_epochs']}")
    print(f"Learning Rate: {cfg['train']['lr']}")
    print(f"Optimizer: {cfg['train']['optimizer']}")
    print(f"Weight Decay: {cfg['train']['weight_decay']}")
    print(f"Batch Size: {cfg['train']['batch_size']}\n")


    backbone, backbone_transform = create_model(model_name = cfg['model']['backbone'], 
                                                fine_tune = False,
                                                num_classes = cfg['model']['num_classes'])
    backbone.to(device)
    
    print(f"{backbone.__class__.__name__} Model Summary") 

    summary(model=backbone, 
        input_size=(32, 3, 224, 224),
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
        
    LOG.info(f"Backbone {cfg['model']['backbone']} succesfully loaded.")

    optimizer = get_optimizer(cfg, backbone)
    LOG.info(f"Optimizer has been defined.")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Loss function.
    criterion = nn.CrossEntropyLoss()
    LOG.info(f"Criterion has been defined")

    # initialize SaveBestModel class
    save_best_model = SaveBestModel()
    
    print('Start to preparing datasets')
    
    #1. VehicleMakeModelDataset
#     dataset = VehicleMakeModelDataset(cfg)

    # 2. VehicleColorDataset
    dataset = VehicleColorDataset(cfg)

    train_dl = dataset.train_dataloader()
    val_dl = dataset.val_dataloader()
    test_dl = dataset.test_dataloader()
    LOG.info(f"Dataset train, val, and test data loader successfully loaded.")

    logger.log_parameters(hyperparameters)
    LOG.info("Parameters has been Logged")

    generate_model_config(cfg)
    LOG.info("Model config has been generated")

    set_seeds()
    
    train(model = backbone, 
          train_loader = train_dl, 
          val_loader = val_dl,
          optimizer = optimizer, 
          criterion = criterion, 
          device = device, 
          cfg = cfg)
          
    save_dir = cfg['output_dir'] + cfg.model.backbone

    best_model_path = os.path.join(save_dir, f'best_model_{cfg.model.backbone}.pth')
    final_model_path = os.path.join(save_dir, f'final_model_{cfg.model.backbone}.pth')
    model_cfg_path = os.path.join(save_dir, f'model-config-{cfg.model.backbone}.yaml')
    acc_fig = os.path.join(save_dir, f'acc_figure_{cfg.model.backbone}.png')
    loss_fig = os.path.join(save_dir, f'loss_figure_{cfg.model.backbone}.png')
    
    artifact.add(best_model_path)
    artifact.add(final_model_path)
    artifact.add(model_cfg_path)
    artifact.add(acc_fig)
    artifact.add(loss_fig)

    logger.log_artifact(artifact=artifact)
    
    logger.end()