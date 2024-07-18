#!/bin/bash

while true; do
    # Run the script with a timeout of 12 hours (43,200 seconds)
    timeout 14400 python -u download_pipeline_parallel.py >> download.log 2>&1

    # Check the exit status of the script
    if [ $? -ne 0 ]; then
        echo "Script crashed. Restarting in 5 seconds..." >> download4.log
        sleep 5
    else
        echo "Script completed successfully."
        break
    fi
done