#!/bin/bash
USERNAME=
HOST=
JUPYTER_PORT=12345
SSH_KEY_PATH=              # for Mobaxterm: /drives/c/...
INIT_COMMANDS="source activate ...; cd ..."

ssh $USERNAME@$HOST -tx -L $JUPYTER_PORT:127.0.0.1:$JUPYTER_PORT -i $SSH_KEY_PATH screen -dR -S jupyter "bash -c '$INIT_COMMANDS; jupyter notebook --port $JUPYTER_PORT --no-browser'"

# -t: terminal; -x: disable X11
# -dR will attach to a unique (detach first if needed) screen session -  or use the other arguments to create a new one
# -S named screen session
# check screen sessions with "screen -list"
# you can only attach to detached screen sessions

# if SSH connection break, you may end up with a session which not deattached (in that case screen will erroneously start new sessions)
# you can log in to the server and use "screen -d ..." (after "scree -list") to detach the session
