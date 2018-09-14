#!/bin/bash
ssh anton@REMOTE -tt -L 12345:localhost:12345 -i DESTHOME/.ssh/ed25519 'source activate py3 && jupyter notebook --port 12345 --no-browser'
# without -tt it will not kill jupyter on disconnect
