import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.system('py -m visdom.server')


cmd1 = 'python train.py --dataroot ./datasets/truncationData --name truncationPix2pix  --model pix2pix --dataset_mode aligned --display_id 0 --preprocess none --num_threads 0 --gpu_ids 0 --batch_size 10 --input_nc 1 --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --netG unet_256 --isTiff --norm batch --pool_size 0'


cmd2 = 'python test.py --dataroot ./datasets/truncationData --name truncationPix2pix  --model pix2pix --dataset_mode aligned --preprocess none --direction AtoB --input_nc 1 --output_nc 1 --netG unet_256  --isTiff --norm batch'


cmds = [cmd1, cmd2]

for cmd in cmds:
    os.system(cmd)



