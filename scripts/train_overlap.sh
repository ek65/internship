for i in {1..9..1}
do
  sh /home/tommaso/squeezeDet/scripts/train.sh -data_path=/home/tommaso/scenicEx/data/matrix/ -train_set=train_matrix_over_${i} -train_dir=/home/tommaso/scenicEx/data/matrix/checkpoints/train_matrix_over_${i}
done
