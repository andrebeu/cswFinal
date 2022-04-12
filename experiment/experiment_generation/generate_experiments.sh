nsubs=$1
npasses=$2 

# in experiment_generation/exp_temp
# copy the contents of mturk_template_folder 
# use the cswEngine to generate task scripts in exp_temp
# rsync from VM to local, and copy most recent databse into exp_temp
# rsync the contents of exp_temp into VM

# directories and folders
csw_dir="/Users/abeukers/wd/csw/cswBehavioral"

# make new experiment folder
exp_dir="${csw_dir}/experiment_generation/exp_temp"
rm -r "${exp_dir}/"* # ensure it is clean

# copy cotents of mturk_template_folder into exp_dir
cp -r "${csw_dir}/experiment_generation/csw_mturk_template/"* ${exp_dir}

# echo "generate task scripts" ##
cd "${csw_dir}/cswEngine"
# run tests
python "cswTests.py"
# generate N scripts with k passes in ${csw_dir}/experiment_generation/task_scripts/
python -c "import cswMturk; cswMturk.write_N_mturk_taskscripts(${nsubs},${npasses})"

# echo "move task scripts into ${exp_dir}/static/js/task" ##
task_folder="${exp_dir}/static/js/task"
mkdir -p ${task_folder} 
mv "${csw_dir}/experiment_generation/task_scripts_temp/"* "${task_folder}"

# echo "copy cswEngine into task folder" ##
cp -r "${csw_dir}/cswEngine/"* "${exp_dir}/static/cswEngine/"

# echo "copy most up-to-date version of database into experiment folder" ##
bash "../experiment_data/rsync_top_down.sh"

echo "copy global database into experiment folder"
cp "${csw_dir}/csw_mturk_spring19.db" "${exp_dir}"

## copy experiment folder into VM
printf "rsync experiment folder into VM\n"
rsync -r -vam --update "${exp_dir}/"* \
						"psiturk-abeukers.princeton.edu:/home/abeukers/csw/exp_$(date +%y%m%d%H%M%S)"

rm -r "${exp_dir}/"* # clear out exp_temp