#/usr/local/bin/mpirun -np 2 ../../../build/tools/caffe train \
#    --solver=./Google_solver.prototxt 2>&1 | tee G_D1_s1.txt
postfix=`date +"%m-%d-%y"`

/usr/bin/mpirun -np 4 /home/czhang/wyang/caffe_mpi/build/install/bin/caffe train \
--weights=/home/czhang/wyang/model/official_bbox_ctx_32_finetuning_iter_100000.caffemodel  \
--solver=./solver_freeze_bn.prototxt 2>&1 | tee log-$postfix.txt

