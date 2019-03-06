#!/bin/bash
USERNAME=
HOST=
JUPYTER_PORT=12345
SSH_KEY_PATH=              # for Mobaxterm: /drives/c/...
INIT_COMMANDS="source activate ...; cd ..."

ssh $USERNAME@$HOST -tx -L $JUPYTER_PORT:127.0.0.1:$JUPYTER_PORT -i $SSH_KEY_PATH screen -R -S jupyter "bash -c '$INIT_COMMANDS; jupyter notebook --port $JUPYTER_PORT --no-browser'"

# -t: terminal; -x: disable X11
# -R will attach to a unique (detached) screen session -  or use the other arguments to create a new one
# -S named screen session
# check screen sessions with "screen -list"
# you can only attach to detached screen sessions
