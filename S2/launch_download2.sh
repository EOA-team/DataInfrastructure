#!/bin/bash

while true; do
    python -u download_pipeline_parallel2.py >> download2.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Script crashed. Restarting in 5 seconds..." >> download2.log
        sleep 5
    else
        break
    fi
done
