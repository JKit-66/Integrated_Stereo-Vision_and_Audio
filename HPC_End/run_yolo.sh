#!/bin/bash
####################################
#SBATCH --job-name=first_job      # Job name
#SBATCH --output=output_job.out    # Standard output and error log
#SBATCH --error=example_job.err     # Error log
#SBATCH --ntasks=1                  # Request 1 task (since you have 1 CPU)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jkl1a20@soton.ac.uk


log_file="job_times.log"
start_time=$(date +%s)


EXECDIR=/home/jkl1a20/snap/snapd-desktop-integration/157/Desktop/projectfolder/roboflow
WORKDIR=/home/jkl1a20/snap/snapd-desktop-integration/157/Desktop/projectfolder/results
PYTHNENV=/home/jkl1a20/snap/snapd-desktop-integration/157/Desktop/pythonfile

module load python/3.10.12
module load conda
module load ultralytics
module load matplotlib


cd $PYTHNENV
source newpyenv/bin/activate
cd $EXECDIR
python3 trainYOLO.py

end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$(($end_time - $start_time))
echo "Total running time: $elapsed_time" >> "$log_file"