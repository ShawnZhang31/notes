# Linux安装CMake

1. 下载CMake源码: https://cmake.org/download/
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz
```
2. 从源码编译和安装
```bash
tar zxvf cmake-3.*
cd cmake-3.*
./bootstrap --prefix=/usr/local
make -j$(nproc)
make install
```
3. 验证是否安装成功
```bash
cmake --version
```