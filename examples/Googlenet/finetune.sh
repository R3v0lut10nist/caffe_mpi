

#/usr/local/bin/mpirun -np 2 ../../../build/tools/caffe train \
#    --solver=./Google_solver.prototxt 2>&1 | tee G_D1_s1.txt

/usr/bin/mpirun -np 4 /home/czhang/caffe_yjxiong_new/caffe_yjxiong/build/install/bin/caffe train \
--weights=./pre_train.caffemodel  \
    --solver=./solver_freeze_bn.prototxt 2>&1 | tee log_new_1.txt

