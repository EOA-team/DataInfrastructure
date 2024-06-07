#!/bin/bash

while true; do
    python -u download_pipeline_parallel.py >> download.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Script crashed. Restarting in 5 seconds..." >> download.log
        sleep 5
    else
        break
    fi
done
