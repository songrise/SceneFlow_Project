make -C src/tf_ops/3d_interpolation clean 
make -C src/tf_ops/3d_interpolation ARCHI=$1
make -C src/tf_ops/grouping clean 
make -C src/tf_ops/grouping ARCHI=$1
make -C src/tf_ops/sampling clean
make -C src/tf_ops/sampling ARCHI=$1
