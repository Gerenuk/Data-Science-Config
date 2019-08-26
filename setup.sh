#!/bin/bash

EMAIL="a.suchaneck@gmail.com"
USERNAME="Anton Suchaneck"

JUPYTERLAB_EXTENSIONS="@pyviz/jupyterlab_pyviz @jupyter-widgets/jupyterlab-manager @jupyterlab/toc @krassowski/jupyterlab_go_to_definition"
# validate all

# "nomkl" for Linux and AMD to use OpenBLAS (MKL AMD may be slow and even crash, e.g. LGBM)
# nodejs for jupyterlab extensions
# mypy rope for vscode
# toml, click, appdirs for black
CONDA_PACKAGES="ipykernel pandas scikit-learn seaborn holoviews bokeh numexpr bottleneck cytoolz xlrd tqdm statsmodels numpy matplotlib nodejs mypy rope toml click appdirs"
PIP_PACKAGES="colorful blackcellmagic"  # colorlog?

HOSTNAME=`hostname`
SSH_KEY_FILENAME=~/.ssh/${USER}_$HOSTNAME

# Install apt packages
while read package; do
echo "sudo apt install --assume-yes $package"
done <<END
htop
p7zip-full
END

# Add bashrc. Note that some things would need to go top
cp mybashrc ~/.mybashrc
cat <<END >> ~/.bashrc
source ~/.mybashrc
END

# Generate own key SSH key
ssh-keygen -t ed25519 -a 100 -f $SSH_KEY_FILENAME -N ''

# Git config
git config --global user.email $EMAIL
git config --global user.name $USERNAME
git config --global push.default simple                       # New in Git 2.0

# Some command line configs. htop, inputrc
mkdir -p ~/.config/htop
cp htoprc ~/.config/htop/htoprc

cp inputrc ~/.inputrc

# Conda env
conda create -n py -y python=3 $CONDA_PACKAGES

# conda install -c pytorch pytorch

conda activate py
python -m ipykernel install --user --name py --display-name="Py"

pip install $PIP_PACKAGES

jupyter labextension install $JUPYTERLAB_EXTENSIONS

jupyter labextension install @ryantam626/jupyterlab_code_formatter
pip install jupyterlab_code_formatter
jupyter serverextension enable --py jupyterlab_code_formatter

# still need nbextensions

# Jupyter
ipython profile create
jupyter notebook --generate-config
jupyter contrib nbextension install --user

# IPython
ln -s -t ~/.ipython/profile_default ~/config/IPython/startup

#TODO: link snippets.json, but for JupyterLab

# https://github.com/ogham/exa/releases/download/v0.8.0/exa-linux-x86_64-0.8.0.zip
# ...

echo "Your new public SSH key is:"
cat $SSH_KEY_FILENAME.pub
echo "Add this to your Git account and git clone ssh://..."
