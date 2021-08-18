#!/bin/bash

SOURCEPATH="/Users/tacobakker/Downloads/animals/animals/ANIMALS"
DATAPATH="/Users/tacobakker/machinelearning/data/testwild"
echo "[INFO] Getting data from $SOURCEPATH"
echo "[INFO] Installing on $DATAPATH"

echo "[INFO] Creating train and test folders"
mkdir -p $DATAPATH/{train,test}

for d in $SOURCEPATH/*/; do
   NAME=$(basename "$d")
   NUM_FILES=$(ls -v $SOURCEPATH/$NAME | wc -l)
   echo "[INFO] Found $NAME"
   if (( NUM_FILES < 100 )); then
     echo "[WARN] Low amount of training data."
   fi
   echo "[INFO] Copying $NUM_FILES pictures to $NAME train folder"
   cp -R $SOURCEPATH/$NAME/ $DATAPATH/train/$NAME
   cd $DATAPATH/train/$NAME
   echo "[INFO] Renaming $NUM_FILES $NAME files"
   ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done 
   mkdir $DATAPATH/test/$NAME
   cd $DATAPATH/train/$NAME
   echo "[INFO] Moving 5 files from $NAME train to $NAME test folder"
   ls -v | head -n5 | xargs -J X mv X $DATAPATH/test/$NAME
done

exit 0

