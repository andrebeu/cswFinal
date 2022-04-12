#!/bin/bash


# Declare an array of string with type
declare -a StringArray=("interleaved" "interleaved_rep" "blocked" "blocked_rep" "explicit_interleaved" "inserted_early" "inserted_early_rep" "inserted_middle" "inserted_middle_rep" "inserted_late" "inserted_late_rep")
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   python all_plts.py $val
done