#!/bin/bash
rm -rf ./results1/*
rm -rf ./results2/*
rm -rf ./results3/*

(nohup python main.py --img "/home/tmc/Documents/Data/tulagu/1MOV" \
    --out "/home/tmc/project/MidasDepth/results1/" \
    --model 1 >> nohup_model1.out &) && 

(nohup python main.py --img "/home/tmc/Documents/Data/tulagu/1MOV" \
    --out "/home/tmc/project/MidasDepth/results2/" \
    --model 2 >> nohup_model2.out &) &&

(nohup python main.py --img "/home/tmc/Documents/Data/tulagu/1MOV" \
    --out "/home/tmc/project/MidasDepth/results3/" \
    --model 3 >> nohup_model3.out &) 
