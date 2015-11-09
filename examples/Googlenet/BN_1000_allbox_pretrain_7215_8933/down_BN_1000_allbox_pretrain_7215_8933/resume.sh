#/usr/local/bin/mpirun -np 2 ../../../build/tools/caffe train \
#    --solver=./Google_solver.prototxt 2>&1 | tee G_D1_s1.txt
postfix=`date +"%m-%d-%y"`

/usr/bin/mpirun -np 4 /home/czhang/wyang/caffe_mpi/build/install/bin/caffe train \
--snapshot=BN_box_pad16_freeze_bn_dropout_iter_80000.solverstate  \
--solver=./solver_freeze_bn.prototxt 2>&1 | tee log-resume-$postfix.txt

