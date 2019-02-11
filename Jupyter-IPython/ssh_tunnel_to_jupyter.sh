USERNAME=asuchaneck
HOST=11.217.46.194
JUPYTER_PORT=12345
CONDA_ENV_NAME=py
SSH_KEY_PATH=c/Users/VWT737G/.ssh/ed25519

# Your remote ~/.bashrc needs to include the directory of `activate` (from anaconda) unless you specify the full path below
# e.g. export PATH=$PATH:..../Anaconda/bin
# Note that standard .bashrc may exit/return at the top for non-interactive logins -> put your export before that

ssh $USERNAME@$HOST -tt -L $JUPYTER_PORT:127.0.0.1:$JUPYTER_PORT -i /drives/$SSH_KEY_PATH "source activate $CONDA_ENV_NAME; jupyter notebook --port $JUPYTER_PORT --no-browser"

# without -tt it will not kill jupyter on disconnect
