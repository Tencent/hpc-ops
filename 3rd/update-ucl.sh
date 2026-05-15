set -xe

rm -rf ucl
rm -rf ucl.git

git clone --recursive git@git.woa.com:astral/OpenUCL/ucl-hpc.git ucl.git
git -C ./ucl.git checkout 39589ced

mkdir -p ucl
mv ucl.git/cmake ucl/
mv ucl.git/src ucl/
mv ucl.git/CMakeLists.txt ucl/

mkdir -p ucl/nvshmem
mv ucl.git/nvshmem/src   ucl/nvshmem/
mv ucl.git/nvshmem/CMakeLists.txt ucl/nvshmem/
mv ucl.git/nvshmem/License.txt ucl/nvshmem/
mv ucl.git/nvshmem/cmake_config ucl/nvshmem/
mv ucl.git/nvshmem/pkg ucl/nvshmem/
mv ucl.git/nvshmem/*.sym ucl/nvshmem/

rm -rf ucl.git