#!/bin/bash


for SS in {10,100,1000,10000}; do
    python3 main.py "${SS}" &
done


