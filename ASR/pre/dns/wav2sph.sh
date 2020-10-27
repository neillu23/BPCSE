#sox -t wav file.wav -t sph ofile.sph
#training data is already sph 
noisy_path="/home/neillu/Desktop/Workspace/DNS-Challenge-227/DNS-Challenge/training/noisy/"
target_path="/home/neillu/Desktop/Workspace/DNS-Challenge-227/DNS-Challenge/training/noisy_sph/"

rm -r $target_path
mkdir $target_path
for noisy_file in $noisy_path/*.wav; do
    name=$(echo $noisy_file | rev | cut -d "/" -f 1| rev)
    sox -t wav "$noisy_file" -t sph "$target_path/$name"
done
