from easydict import EasyDict as edict
from module.class_names import class_names
from torchvision import transforms
from torch.nn import functional as F
from module.models import create_model
from module.utils import metrics_report_to_df, save_plot_cm, prec_score, \
                        recc_score, acc_score, fone_score, classification_reports, set_seeds
from pycm import *
from matplotlib import pyplot as plt
import os
import time
import torch
import numpy as np
import yaml
import glob as glob
import cv2
import argparse


def transform(image):
    transformation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
    return transformation(image)

def main(args):
    init = set_seeds()
    device = torch.device('cuda:0') if torch.cuda.is_available() \
              else torch.device('cpu')
    print(f"Computation device: {device}")
    print(f"Backbone model: {args.model}")
    
    model_path = f"../Trainer/outputs/{args.model}/model-config-{args.model}.yaml" 
    
    cfg = edict(yaml.safe_load(open(model_path, "r")))
    print(cfg)
    
    model = create_model(model_name = cfg['model'], 
                        num_classes = cfg['output_class']).to(device)
    
    model.eval()

    pretrained_path = os.path.join(f'../Trainer/outputs/{args.model}/', 
                                    cfg.model_file)

    print(model.load_state_dict(torch.load(pretrained_path, 
                                    map_location=device)['model_state_dict'], 
                                    strict=False))
    
    test_images = glob.glob(args.testdir, recursive=True)
    #i=0
    y_true = []
    y_pred  = []
    t0 = time.time()
    for image_path in test_images:
        # if i == args.numsample:
        #   break
        # i += 1
        image = cv2.imread(image_path)
        actual_class_name = image_path.split(os.path.sep)[-2]
        original_image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = original_image.shape

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        start_time = time.time()

        with torch.no_grad():      
            outputs = model(image_tensor)
      
        end_time = time.time()

        probs = F.softmax(outputs, dim=1)
        conf, classes = torch.max(probs, 1)
        conf_score = float(conf.item())
        class_idx = int(classes.item())
        pred_class_name = str(class_names[int(class_idx)])

        y_true.append(actual_class_name)
        y_pred.append(pred_class_name)
        
        print(f"Actual: {actual_class_name} / Prediction: {pred_class_name} ({conf_score})")
        print(f"Inference Time: {(end_time - start_time)*1000 :.3f} ms.")
      
    t1 = time.time()
    print(f"Time taken: {(t1 - t0)*1000} ms.")

    if args.savemetrics:  
        classification_report = metrics_report_to_df(y_true, y_pred, args.savedir)
        save_plot_cm(y_true, y_pred, args.savedir)
        print(f"Precision score (weighted): {prec_score(y_true, y_pred, class_names):.3f}")
        print(f"Recall score (weighted): {recc_score(y_true, y_pred, class_names):.3f}")
        print(f"Accuracy score (weighted): {acc_score(y_true, y_pred):.3f}")
        print(f"F1 Score score (weighted): {fone_score(y_true, y_pred, class_names):.3f}")
        print(f"Clssification report:\n")
        print(classification_reports(y_true, y_pred, class_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument for Inference")
    parser.add_argument('-model', '--model', default='efficientnet_b1',
                        type=str, help="Model is used for inference")

    parser.add_argument('-td', '--testdir', 
                        default='../Dataset/test/*/*.jpg',
                        type=str , help="Path to test directory" )
    
    parser.add_argument('-n', '--numsample', 
                        default=8041, type=int , 
                        help="Number of sample of test directory")
    
    parser.add_argument('-sm', '--savemetrics',
                        default=True, type=bool, 
                        help="Save metrics evaluation")
    
    parser.add_argument('-sd', '--savedir',
                        default='../Trainer/outputs/efficientnet_b1/', type=str, 
                        help="Path to save metrics evaluation")
    
    args = parser.parse_args()
    main(args)