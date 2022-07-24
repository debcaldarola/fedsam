#!/usr/bin/env bash

pushd ../models

declare -a alphas=("0" "0.05")

function run_fedavg() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}

function run_fedavg_with_swa() {
  echo "############################################## Running FedAvg with SWA ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha} -swa --swa-c 10 --swa-start 7500
}

function run_fedsam() {
  echo "############################################## Running FedSAM ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm sam -rho 0.1 -eta 0
}

function run_fedsam_with_swa() {
  echo "############################################## Running FedSAM with SWA ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm sam -rho 0.1 -eta 0 -swa --swa-c 10
}

function run_fedasam() {
  echo "############################################## Running FedASAM ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm asam -rho 0.7 -eta 0.2
}

function run_fedasam_with_swa() {
  echo "############################################## Running FedASAM with SWA ##############################################"
  alpha="$1"
  python main.py -dataset cifar10 --num-rounds 10000 --eval-every 100 --batch-size 64 --num-epochs 1 --clients-per-round 5 -model cnn -lr 0.01 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm asam -rho 0.7 -eta 0.2 -swa --swa-c 10
}

echo "####################### EXPERIMENTS ON CIFAR10 #######################"
for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_fedavg "${alpha}"
  run_fedavg_with_swa "${alpha}"
  run_fedsam "${alpha}"
  run_fedsam_with_swa "${alpha}"
  run_fedasam "${alpha}"
  run_fedasam_with_swa "${alpha}"
  echo "Done"
done