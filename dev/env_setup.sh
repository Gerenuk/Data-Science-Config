#!/bin/bash

# Container setup (call this again on any new instance)
~/config/container_setup.sh

cat <<END >> .bashrc
source ~/config/Linux/bashrc
END

# SSH key
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519 -N ''

# Git config
git config --global user.email "a.suchaneck@gmail.com"
git config --global user.name "Anton Suchaneck"
git config --global push.default simple                       # New in Git 2.0

# Some command line configs
mkdir -p ~/.config/htop
cp ~/config/Linux/htoprc ~/.config/htop/htoprc
cp ~/config/Linux/inputrc ~/.inputrc

# Conda env
conda create -n py3 -y python=3 ipykernel
source activate py3
python -m ipykernel install --user --name py3 --display-name="Py3 (env)"
conda install -y pandas scikit-learn seaborn holoviews bokeh cytoolz xlrd

# Jupyter
# nbextension already installed from container_setup.sh
#!!!! works only after installing it
ipython profile create
jupyter notebook --generate-config
jupyter contrib nbextension install --user

# IPython
mkdir -p ~/.ipython/profile_default/startup/
ln -s -t ~/.ipython/profile_default ~/config/IPython/startup

# https://github.com/ogham/exa/releases/download/v0.8.0/exa-linux-x86_64-0.8.0.zip
# ...

# Setup project
mkdir ~/Projects
cd ~/Projects

echo "Your new public SSH key is:"
cat ~/.ssh/id_ed25519.pub
echo "Add this to your Git account and git clone ssh://..."
