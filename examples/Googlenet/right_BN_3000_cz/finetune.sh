

#/usr/local/bin/mpirun -np 2 ../../../build/tools/caffe train \
#    --solver=./Google_solver.prototxt 2>&1 | tee G_D1_s1.txt

/usr/bin/mpirun -np 4 /home/czhang/run_rcnn/caffe-mpi_parallel_v3/build/install/bin/caffe train \
--weights=./3k_bbox_bn.caffemodel  \
    --solver=./solver_freeze_bn.prototxt 2>&1 | tee log_new_1.txt

