'''
Inference Script for Running Difference Object Detectors from MMDETECTION.

@author shefalis on 3/4/22.
'''

from mmdet.apis import init_detector, inference_detector
import argparse
import os
import mmcv
import cv2
import numpy as np

def run_inference(
    config_file, 
    checkpoint_file, 
    data_path, 
    output_directory,
    cuda_device
):
    '''
    Run Inference on the provided frames given the configuration of the corresponding detector.

    -- config_file: Configuration file from mmdetection repository of the object detector.
    -- checkpoint_file: Checkpoint file from mmdetection repository of the object detector model.
    -- data_path: Path of the data frames.
    -- output_directory: Output directory to save the inference results.
    -- cuda_device: Device number of the GPU to use. 
    '''

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=f'cuda:{cuda_device}')

    # test a list of frames and save the results
    dir_list = os.listdir(data_path)
    for i, frame in enumerate(dir_list):

        # Frame Path
        frame_path = os.path.join(data_path, frame)

        # Input Image
        input_image = cv2.imread(frame_path)

        # Output Image
        result = inference_detector(model, frame_path)
        output_image = np.array(model.show_result(frame_path, result))

        # Save Side by Side Plots.
        combined_image = np.concatenate([input_image, output_image], axis=1)
        output_path = os.path.join(output_directory, f'{frame}')
        cv2.imwrite(output_path, combined_image)



if __name__ == "__main__":

    # python inference.py --config_file name_of_config_file --checkpoint_file name_of_checkpoint_file> --data_path data_path --output_directory output_directory --cuda_device cuda_device_number
    # python inference.py --config_file 'configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-71e2215e.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment1' --cuda_device 0   
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-a698dd3d.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment2' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-a1c11314.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment3' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-a0b59c42.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment4' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-392a804b.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment5' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-cd0f6a12.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment6' --cuda_device 0 
    
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-e68eb464.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment7' --cuda_device 0 
    
    # python inference.py --config_file 'configs/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-1d817139.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment8' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-8b5a6745.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment9' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_mstrain_2x_lvis_v1-5d8ca2a4.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment10' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1-71e2215e.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment11' --cuda_device 0 
    # python inference.py --config_file 'configs/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py' --checkpoint_file 'checkpoints/seesaw_loss/cascade_mask_rcnn_r101_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1-c8551505.pth' --data_path 'data/sample_data/construction_data' --output_directory 'outputs/sample_data/construction_data/experiment12' --cuda_device 0 


    parser = argparse.ArgumentParser(description='Inference Script Arguments')

    parser.add_argument('--config_file', type=str)
    parser.add_argument('--checkpoint_file', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--cuda_device', type=int)

    args = parser.parse_args()

    with open(f'{args.output_directory}/config.txt', 'w') as f:
        f.write(f'config_file: {args.config_file}\n')
        f.write(f'checkpoint_file: {args.checkpoint_file}\n')
        f.write(f'data_path: {args.data_path}\n')
        f.write(f'output_directory: {args.output_directory}\n')
        f.write(f'cuda_device: {args.cuda_device}\n')
        f.close()

    # Run inference with the given arguments.
    run_inference(
        args.config_file, 
        args.checkpoint_file, 
        args.data_path, 
        args.output_directory,
        args.cuda_device
    )
    