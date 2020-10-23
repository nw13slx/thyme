#!/bin/env bash

file=$1
xyz=$2

n=$(grep "Position\|^$" $file -n \
    |grep El -A1\
    |head -n 2 \
    |sed "s/://g" \
    |awk '{if (NR==1) a=$1; else a=$1-a-1}END{print a}')
echo $n

grep -P "\d+\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)\s+([+-]?\d+.\d+[eE]?[+-]?\d*)" $file \
    |awk -v n=$n '{if (NR%n==1) {print n, "\n";} print $1, $2, $3, $4}' > $xyz

