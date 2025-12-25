set -xe

rm -rf cutlass 
rm -rf cutlass.git

git clone https://github.com/NVIDIA/cutlass.git cutlass.git
git -C ./cutlass.git checkout dc481792

mkdir -p cutlass 
mv cutlass.git/include cutlass/

rm -rf cutlass.git
