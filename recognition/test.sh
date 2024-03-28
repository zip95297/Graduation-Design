check_points_root_path="../checkpoints/ckpt-recognition"
for file in "$check_points_root_path"/*; do
# 保证是文件不是文件夹
    if [ -f "$file" ]; then
        tmp=${file##*/}
        arg=${tmp%.*}
        python test.py $arg >> test_result
        echo "$file"
        mv -n $file ../checkpoints/ckpt-recognition/Tested
        echo "done"
    fi
done
# nohup ./test.sh  &