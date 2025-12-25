set -xe

rm -rf nvshmem
rm -rf nvshmem.git

git clone git@git.woa.com:astral/trmt/trmt-shmem.git nvshmem.git
git -C ./nvshmem.git checkout 78704b40

mkdir -p nvshmem
mv nvshmem.git/src   nvshmem/
mv nvshmem.git/CMakeLists.txt nvshmem/
mv nvshmem.git/License.txt nvshmem/
mv nvshmem.git/changelog nvshmem/
mv nvshmem.git/cmake_config nvshmem/
mv nvshmem.git/pkg nvshmem/
mv nvshmem.git/*.sym nvshmem/

rm -rf nvshmem.git
