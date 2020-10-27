#move train_noisy to TRAIN 
#move test_noisy_sph to TEST 

noisy_path="~/Downloads/TIMIT_wav_noisy/testing/test_noisy_sph"
clean_path="~/Downloads/TIMIT_Neil/TIMIT/TEST"

# noisy_path="~/Downloads/TIMIT_wav_noisy/training/train_noisy"
# clean_path="~/Downloads/TIMIT_Neil/TIMIT/TRAIN"

for noisy_file in $noisy_path/*.wav; do
    noisy_name=$(echo $noisy_file | rev | cut -d "/" -f 1| rev)
    folder=$(echo $noisy_name | cut -d "_" -f 1)
    folder=${folder^^}
    name=$(echo $noisy_name | cut -d "_" -f 2)
    name="${name^^}.WAV"
    # mkdir -p $clean_path/$folder
    #echo "cp $noisy_file $clean_path/$folder/$name"
    # echo $folder,$name
    for dir in $clean_path/*; do
        if [ -d $dir/$folder ]; then
            if [ -f $dir/$folder/$name ]; then
                # echo "copy $noisy_file to exists $dir/$folder/$name."
                cp "$noisy_file" $dir/$folder/$name
            else
                echo "File $dir/$folder/$name does not exists."
            fi
        fi
    done
done
