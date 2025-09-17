提交：scp /d/HOMEWORK/3710/3710-3/cifar10_dawnbench/train_cifar10.py \
    s4908583@rangpur.compute.eait.uq.edu.au:~/cifar10_dawnbench/
scp /d/HOMEWORK/3710/3710-3/cifar10_dawnbench/train_cifar10.sh \
    s4908583@rangpur.compute.eait.uq.edu.au:~/cifar10_dawnbench/
提交2：sbatch train_cifar10.sh
队列：squeue -u s4908583
查看状态：scontrol show job 
查看输出：tail -f ~/cifar10_dawnbench/cifar10.out
tail -f ~/unet_project/unet.out
查看错误：tail -f ~/cifar10_dawnbench/cifar10.err
cd：cd ~/unet_project
cd ~/cifar10_dawnbench
激活环境：conda activate torch_3710_py310
连接到学校：ssh s4908583@rangpur.compute.eait.uq.edu.au	
