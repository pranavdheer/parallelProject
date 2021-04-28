#!/bin/bash

TESTDATA="http://snap.stanford.edu/data/as-skitter.txt.gz"

g++ -std=c++11 -O3 convert-from-snap-main.cpp -o convert-from-snap-main.o

mkdir -p data
cd data


base=`echo $url | grep -o "[^/]\+$" | sed "s/\(\.txt\.gz\|\.graph\.bz2\)$//"`
if [ -f ${base}.bin ]
then
  echo Skipping $base because it already exists
  continue
fi
wget $url
echo Unzipping and converting

gzip -d ${base}.txt.gz
../convert-from-snap-main.o ${base}.txt ${base}.bin
rm ${base}.txt
