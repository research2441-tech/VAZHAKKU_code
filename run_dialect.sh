#!/bin/bash
# usage: run_dialect.sh NAME BACKBONE [extra args...]   -> trains then logs
cd "$(dirname "$0")"; L=${VZ_ROOT:-.}/work/logs; C=${VZ_ROOT:-.}/work/ckpt
NAME=$1; BB=$2; shift 2
[ -f $C/$NAME.pt ] && { echo "skip $NAME"; exit 0; }
python3 train_dialect.py --backbone $BB --out $C/$NAME.pt --steps ${DS_STEPS:-2000} --E 4 --Q 3 --K 5 "$@" > $L/$NAME.log 2>&1
