#sox -t wav file.wav -t sph ofile.sph
#training data is already sph 
noisy_path="TIMIT_wav_noisy/testing/test_noisy"
target_path="TIMIT_wav_noisy/testing/test_noisy_sph"

rm -r $target_path
mkdir $target_path
for noisy_file in $noisy_path/*.wav; do
    name=$(echo $noisy_file | rev | cut -d "/" -f 1| rev)
    name1=$(echo $name | cut -d "_" -f 1)
    name_dash="_"
    name2=$(echo $name | cut -d "_" -f 2)
    sox -t wav "$noisy_file" -t sph "$target_path/$name1$name_dash$name2$name_dash.wav"
done
