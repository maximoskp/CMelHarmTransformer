#!/bin/bash

# List of Python scripts with their respective arguments
# beam search 5 CAtest
# scripts=(
#     "generate_gpt_beam.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 7 -p 0.0 -b 1"
#     "generate_gpt_beam.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 7 -p 0.0 -b 1"
#     "generate_bart_beam.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 7 -p 0.0 -b 1"
#     "generate_bart_beam.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 7 -p 0.0 -b 1"
# )

# scripts=(
#     "generate_tokbytok_gpt.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -b 1"
#     "generate_tokbytok_gpt.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -b 1"
#     "generate_tokbytok_bart.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -b 1"
#     "generate_tokbytok_bart.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -b 1"
# )

# astar
# scripts=(
#     "generate_gpt_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 4 -l 4 -b 1"
#     "generate_gpt_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 4 -l 4 -b 1"
#     "generate_bart_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 4 -l 4 -b 1"
#     "generate_bart_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 4 -l 4 -b 1"
# )
# scripts=(
#     "generate_gpt_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 50 -l 25 -b 1"
#     "generate_gpt_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 50 -l 25 -b 1"
#     "generate_bart_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 50 -l 25 -b 1"
#     "generate_bart_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 50 -l 25 -b 1"
# )
scripts=(
    "generate_gpt_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 10 -l 5 -b 1"
    "generate_gpt_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 0 -s 10 -l 5 -b 1"
    "generate_bart_astar.py -t ChordSymbolTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 10 -l 5 -b 1"
    "generate_bart_astar.py -t PitchClassTokenizer -v /media/maindisk/maximos/data/hooktheory_test -g 1 -s 10 -l 5 -b 1"
)

# Name of the conda environment
conda_env="torch"

# Loop through the scripts and create a screen for each
for script in "${scripts[@]}"; do
    # Extract the base name of the script (first word) to use as the screen name
    screen_name=$(basename "$(echo $script | awk '{print $1}')" .py)
    
    # Start a new detached screen and execute commands
    screen -dmS "$screen_name" bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh;  # Update this path if your conda is located elsewhere
        conda activate $conda_env;
        python $script;
        exec bash
    "
    echo "Started screen '$screen_name' for script '$script'."
done
