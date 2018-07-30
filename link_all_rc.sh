#!/bin/bash
ln -s ~/.inputrc inputrc

touch ~/.bashrc
cat >> ~/.bashrc <<END
source "`readlink -f mybashrc`"
END
