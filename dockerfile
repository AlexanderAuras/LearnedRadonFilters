#Setup OS
FROM debian:11
SHELL ["/bin/bash", "-c"]
RUN apt-get update

#Setup user
RUN apt-get install -y sudo;\
    echo "lrf_user ALL=(ALL) ALL" >> /etc/sudoers;\
    useradd -m -s /bin/bash lrf_user;\
    echo "lrf_user:lrf_password" | chpasswd
USER lrf_user

#Setup anaconda
RUN echo "lrf_password" | sudo -S apt-get install -y wget;\
    mkdir /home/lrf_user/Downloads;\
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O /home/lrf_user/Downloads/anaconda.sh;\
    bash /home/lrf_user/Downloads/anaconda.sh -b -p /home/lrf_user/anaconda3;\
    rm -rf /home/lrf_user/Downloads;\
    eval "$(/home/lrf_user/anaconda3/bin/conda shell.bash hook)";\
    conda init bash

#Create environment
RUN eval "$(/home/lrf_user/anaconda3/bin/conda shell.bash hook)";\
    conda create --name LRF python=3.10;\
    conda activate LRF;\
    conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7
RUN eval "$(/home/lrf_user/anaconda3/bin/conda shell.bash hook)";\
    conda activate LRF;\
    conda install -y -c anaconda numpy jupyter;\
    conda install -y -c conda-forge matplotlib hydra-core torchmetrics ninja tensorboard
RUN eval "$(/home/lrf_user/anaconda3/bin/conda shell.bash hook)";\
    conda activate LRF;\
	pip install pytorch-lightning;\
    pip install --upgrade hydra_colorlog

#Setup C++/CUDA
RUN echo "lrf_password" | sudo -S apt-get install -y build-essential;\
    echo "lrf_password" | sudo -S ln -s /home/lrf_user/anaconda3/envs/LRF/lib/libcudart.so.11.7.99 /usr/lib/x86_64-linux-gnu/libcudart.so

#Setup git
RUN echo "lrf_password" | sudo -S apt-get install -y git;\
    git config --global user.name "LRF User";\
    git config --global user.email "lrf_user@users.noreply.github.com"

#Clone repositories
RUN eval "$(/home/lrf_user/anaconda3/bin/conda shell.bash hook)";\
    conda activate LRF;\
    git clone https://AlexanderAuras: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@github.com/AlexanderAuras/radon.git /home/lrf_user/radon;\
    cd /home/lrf_user/radon;\
	pip install .;\
    git clone https://AlexanderAuras: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@github.com/AlexanderAuras/LearnedRadonFilters.git /home/lrf_user/LearnedRadonFilters

#Setup project
RUN echo "lrf_password" | sudo -S mkdir -p /data;\
    echo "lrf_password" | sudo -S chmod a+rwx /data;\
    mkdir -p /data/sciebo/experiments/LearnedRadonFilters
EXPOSE 6006
WORKDIR /home/lrf_user/LearnedRadonFilters
CMD bash

#docker run --rm --gpus all -v /data/sciebo/experiments/LearnedRadonFilters:/home/lrf_user/LearnedRadonFilters/runs -p 6006:6006 -t -i lrf
#conda activate LRF
#tensorboard --logdir=/home/lrf_user/LearnedRadonFilters/runs &
#/home/lrf_user/anaconda3/envs/LRF/bin/python /home/lrf_user/LearnedRadonFilters/main.py