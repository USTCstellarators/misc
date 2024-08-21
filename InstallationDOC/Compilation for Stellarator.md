本文档主要引入主流仿星器领域常用套件与代码，并提供编译建议。

[TOC]

#  START

在仿星器位型优化与线圈设计领域存在一系列较为成熟的代码。STELLOPT、SIMSOPT与DESC是三套涵盖了从位型优化到线圈设计的综合优化套件，也是三种现代化程度不同的大型代码工具。FOCUS与REGCOIL是两种常用的三维线圈优化工具，可以不依赖上述套件独立运行。SIMPLE是IPP开发的用于快速计算高能粒子损失的导心运动代码，不依赖上述套件可以独立运行。

对于只需要计算磁流体平衡的情况，上述三个套件提供了两种半的方式：

- VMEC2000

  - STELLOPT中存在VMEC2000软件包，可以直接通过命令运行，`mpirun -n 12 xvmec2000 input.stellarator`

- VMEC2000-python wrap

  - SIMSOPT中提供了VMEC2000的python wrap版本，使用时需要在python文件中通过

    ```python
    from simsopt.mhd import Vmec
    # 初始化 MPI,如果需要并行计算一个input文件，ngroups = 1
    # 需要mpi使用mpirun -n 12 python *.py
    mpi = MpiPartition(ngroups = 1)
    # 定义 VMEC 输入文件路径
    vmec_input_filename = 'input.nfp2_QA'
    # 创建 VMEC 对象并加载输入文件
    vmec = Vmec(vmec_input_filename,mpi=mpi,keep_all_files=True)
    vmec.run()
    # 获取平衡信息
    if mpi.rank_world == 0: 
    	value_xxx = vmec.wout.xxx
        ...
    ```

- DESC
  - DESC是一种通过力平衡求解三维平衡的代码，采用python-jax的方式编写，兼容VMEC输入文件，支持通过命令行运行 `desc input.vmec_input or input.desc_input` 与创建python文件运行。详细方式与区别见后。

## Anaconda

在正式安装套件前，推荐使用conda进行环境管理。conda能够创建虚拟环境以确保环境变量的改变不会导致本机的基础环境发生变化。得益于conda的大规模使用，有关conda的安装方式的介绍已经非常成熟，在此不再赘述。

**详情可参阅：**[Anaconda | The Operating System for AI](https://www.anaconda.com/) / [超详细Ubuntu安装Anaconda步骤+Anconda常用命令_ubuntu 安装anaconda-CSDN博客](https://blog.csdn.net/KRISNAT/article/details/124041869)

不建议在 base 环境下对任何软件进行安装，建议通过下属方式创建名为 your_env_name 的环境： 

```shell
conda create -n your_env_name python=3.xx
```

并通过下述方式激活它

```shell
conda activate your_env_name
```





# STELLOPT

STELLOPT 是由PPPL维护的结合”科学“与”工程“的综合代码。包含了从物理到工程几乎所有与仿星器有关的代码。

**网页：**[VMECwiki | [“STELLOPT”\] (princetonuniversity.github.io)](https://princetonuniversity.github.io/STELLOPT/) 

## **编译**

 STELLOPT的编译教程参见 [STELLOPT Suite Compilation | [“STELLOPT”\] (princetonuniversity.github.io)](https://princetonuniversity.github.io/STELLOPT/STELLOPT Compilation), 该教程不通过conda进行环境管理而通过直接添加系统中软件路径的方式全局编译。

对于Ubuntu，参见 [STELLOPT Compilation on Ubuntu | [“STELLOPT”\] (princetonuniversity.github.io)](https://princetonuniversity.github.io/STELLOPT/STELLOPT Compilation Ubuntu)

1. 首先依据 "STELLOPT Compilation on Ubuntu" 的页面描述，通过 `sudo apt-get install xxxx` 安装所需要的基本库。包括 git/gfortran/openmpi/gfortran/g++/netcdf/blas/lapack/python/scalapack 等

2. 依据 "STELLOPT Compilation on Ubuntu" 的页面描述，设置环境变量：

   ```shell
   export MACHINE="ubuntu"
   export STELLOPT_PATH=<path to repo directory>
   ```

3.  在STELLOPT路径下，通过 `./build_all` 编译全部软件

## 可能遇到的问题

### 1. GCC版本不匹配

例如gcc-11与mpich-4.X的组合会产生严重的类型匹配问题，使得对源码进行严格的类型检查，且不可通过增加忽略标识符 `-fallow-argument-mismatch`  以消除报错（将会导致编译的软件无法正常计算）。正确的解决方法是进行**GCC与GFORTRAN降级**

在已有高版本的gcc/gfortran前提下，通过执行下述命令安装

```bash
sudo apt-get install gcc-9
sudo apt-get install gfortran-9
```

并通过修改优先级的方式调整高低版本编译器的使用顺序

```bash
#调整gcc-11的优先级为40，gcc-9的优先级为100
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 40
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100
 
#调整gfortran-11的优先级为40，gfortran-9的优先级为100
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-11 40
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 100
```

### 2. export环境变量

设置环境变量时键入：

```shell
export MACHINE="ubuntu"
```

对应的行为是在 `./STELLOPT/SHARE/` 路径下对应的 `make_xxx.inc` 配置文件，其中描述了在不同设备上安装所依赖的路径与标识。





# SIMSOPT

SIMSOPT 通过为VMEC等fortran或c++代码包装为python wrap以实现对优化程序的综合调用。

**网页：**[Simsopt documentation — simsopt 1.6.1.dev3+ga1b43f4 documentation](https://simsopt.readthedocs.io/en/latest/)

## 1. General

SIMSOPT的编译可以分为两个部分：SIMSOPT框架与VMEC等计算模块。主框架几乎与同样的python包一样，通过 `pip install .` 的方式即可完成构建。但VMEC等部分则需要通过构建依赖关系以编译。在这里我们建议通过conda方式构建，且此方式可以统一Linux与MacOS的编译。

实际上，该思路来源于[Mac M1 M2 installation · hiddenSymmetries/simsopt Wiki (github.com)](https://github.com/hiddenSymmetries/simsopt/wiki/Mac-M1-M2-installation)，并在多台设备上验证。

## 2. For VMEC2000

在已经创建并激活的conda环境中：

1.  安装 compilers, netcdf-fortran, openmpi-mpicc, openmpi-mpifort, and openblas，同时输入

   ```shell
   conda install compilers netcdf-fortran openmpi-mpicc openmpi-mpifort openblas scalapack
   ```

2. 重启终端，并安装mpi4py, scikit-build, numpy

   ```
   conda install mpi4py scikit-build numpy
   ```

3. 安装 VMEC2000的其它依赖 **(for Ubuntu)**

   ```
   conda install cmake ninja f90wrap
   ```

   安装f90wrap与VMEC2000的其它依赖**（for MacOS）**

   ```
   conda install meson-python
   conda install --no-build-isolation f90wrap
   conda install cmake ninja
   ```

4. 通过[hiddenSymmetries/VMEC2000 (github.com)](https://github.com/hiddensymmetries/vmec2000) 克隆VMEC2000代码库并进入目录。在编译VMEC2000前需要修改 `cmake_config_file.json` . 其模板文件可以在 `../VMEC2000/cmake/machines/` 找到。

   修改该配置文件的主要作用是为cmake添加依赖库的路径，在上述配置后可能编译器无法找到诸如NETCDF、SCALAPACK等编译库的位置，以主动指定，例如

   ```
   "-DCMAKE_C_COMPILER=gcc"，
   "-DCMAKE_CXX_COMPILER=g++",
   "-DCMAKE_Fortran_COMPILER=gfortran",
   "-DSCALAPACK_LIB_NAME=scalapack"，
   "-DNETCDF_INC_PATH=/usr/include",
   "-DNETCDF_LIB_PATH=/usr/lib/x86_64-linux-gnu"
   ```

5. 进行编译

   ```
   python setup.py build_ext
   python setup.py install
   ```

## 3. For Simsopt

在完成上述之后，对于**Ubuntu**：

```shell
conda install -c hiddensymmetries simsopt
```

对于**MacOS**：

1.  安装 LLVM-OpenMP等simsopt依赖项

   ```
   conda install llvm-openmp numpy jax jaxlib scipy Deprecated nptyping monty ruamel.yaml sympy f90nml randomgen pyevtk
   ```

2. 安装依赖项

   ```shell
   conda install cmake ninja setuptools_scm
   ```

3.  从 https://github.com/hiddenSymmetries/simsopt 克隆储存库并进入库中

   ```
   git clone git@github.com:hiddenSymmetries/simsopt.git
   ```

4.  在 储存库中通过一下命令安装

   ```
   pip install --no-build-isolation .
   ```

## 可能遇到的问题

### 1. 找不到netcdf_I/B

在编译VMEC2000时，最常见的问题是找不到netcdf的include与library文件。一般而言，如果上述conda环境安装正确，那么可以直接指定`../usr_name/miniconda3(anaconda3)/envs/your_env_name/`路径下的`include or lib`

### 2. 运行时jax.config报错

由于未知原因在import simsopt时会产生如下错误：` ImportError: cannot import name 'config' from jax.config`

只需要在对应文件目录下修改为 `from jax import config`

### 3. Clang头文件错误

在MacOS，尤其是ARM64构架上的设备进行编译时，由于MacOS采用Clang对C/C++进行编译，因此可能存在部分编译器头文件为更新至与GCC等一致。例如，原始在Darwin内核中某个头文件中

```c++
struct hash_base : std::unary_fuction<T std::size_t> {};
```

其中的`unary_function`需要改为`__unary_function`

```c++
struct hash_base : std::__unary_fuction<T std::size_t> {};
```





# DESC

DESC 使用伪谱数值方法和自动微分求解并优化 3D MHD 平衡。能够在平衡求解上代替VMEC，并可进行多种具有约束的优化。

**网页：**[Stellarator Optimization Package — DESC 0.11.1+641.g0a6b995.dirty documentation (desc-docs.readthedocs.io)](https://desc-docs.readthedocs.io/en/latest/index.html)

DESC 适用于Linux、MacOS、集群等，并且支持x86和ARM64构架，支持CPU加速与GPU加速，并且进行了详尽的包管理。可以简单的通过

```shell
pip install descopt
```

更详细的安装与编译方法可以在说明页找到非常详细的安装说明。不再赘述

**说明网页：**[安装 — DESC 0.11.1+641.g0a6b995.dirty 文档 --- Installation — DESC 0.11.1+641.g0a6b995.dirty documentation (desc-docs.readthedocs.io)](https://desc-docs.readthedocs.io/en/latest/installation.html#)





# FOCUS

Flexible Optimized Coils Using Space curves （FOCUS），通过三维曲线描述环形装置的电磁线圈以产生满足约束的磁场。

FOCUS的编译非常容易，因为具有详细的makefile文件, 编译可以在页面上找到[下载并编译 |重点 --- Download and compile | FOCUS (princetonuniversity.github.io)](https://princetonuniversity.github.io/FOCUS/compile.html)。对于Linux与MacOS都是可行的。

在编译focus前，需要首先具有 Intel/GCC fortran compiler、OpenMPI以及HDF5，在conda中安装这些的方法此前已经有所提及

```shell
conda install gfortran openmpi-mpicc openmpi-mpifort hdf5 netcdf
```

之后通过git 下载源码

```shell
git clone https://github.com/PrincetonUniversity/FOCUS.git
```

进入到FOCUS目录后，直接编译focus

```shell
make CC=gfortran xfocus
```

其中，CC是编译器标识符，可选的还有CC=intel（默认）





# REGCOIL

REGCOIL是一种通过面电流势方法获得线圈形状的代码，可以作为FOCUS的初始值使用。

**地址：**[landreman/regcoil: REGCOIL: A regularized current potential method for rapid and robust computation of stellarator coil shapes (github.com)](https://github.com/landreman/regcoil)

该代码依赖库有：

- 使用MPI的FORTRAN编译器MPIFORT
- 数据库NETCDF、HDF5
- 计算库BLAS以及LAPACK

首先通过conda获得依赖库

```
conda install openmpi-mpifort openblas scalapack netcdf-fortran hdf5
```

之后，通过git获得代码

```shell
git clone https://github.com/landreman/regcoil.git
```

进入文件夹后，找到 `makefile` 文件，可以注意到其makefile中进行预设的编译平台有：cori、marconi、pppl_gcc、pppl_intel、stellar、osx_brew、DRACO、RAVEN 以及 macports（else中），这写平台中除了osx_brew与macports是对于通过homebrew以及macports管理的MacOS系统外，其它均是特定超算的名称。因此无法直接通过指定 `MACHINE =  xxx`的方式编译。实际上，只需要指定fortran编译器`FC = mpifort` 、包含NETCDF、HDF5、BLAS以及LAPCK位置与定义信息`EXTRA_LINK_FLAGS = xxx` 以及编译标识符 `EXTRA_COMPILE_FLAGS = xxx`。综上，可以在makefile第23行开始的判断循环内添加判断条件

```makefile
else ifeq ($(HOSTNAME),Ubuntu)
  REGCOIL_HOST = Ubuntu
  FC = mpifort 
  EXTRA_COMPILE_FLAGS = -O3 -ffree-line-length-none -fopenmp  -fallow-argument-mismatch 
  EXTRA_LINK_FLAGS = -fopenmp -I/home/dmcxe/miniconda3/envs/sims/include/ -L/home/dmcxe/miniconda3/envs/sims/lib/ -lnetcdf -lnetcdff -lhdf5_hl -lhdf5 -lz -lblas -llapack
  
  # For batch systems, set the following variable to the command used to run jobs. This variable is used by 'make test'.
  REGCOIL_COMMAND_TO_SUBMIT_JOB =
```

在上述代码中，`EXTRA_LINK_FLAGS = xxx` 中 `-I/home/dmcxe/miniconda3/envs/sims/include/` 描述的是INCLUDE文件夹的位置，这里只需要将 `-I`后的目录修改为对应环境中的include路径。同理，`-L`描述的是LIB文件夹的位置，只需要将 `-I`后的目录修改为对应环境中的lib路径。

最后，在文件夹中，指定 `MACHINE =  Ubuntu`，并

```shell
make regcoil
```





# SIMPLE

**S**ymplectic **I**ntegration **M**ethods for **P**article **L**oss **E**stimation(SIMPLE)，通过辛积分计算粒子损失。通过VMEC平衡文件计算给定质量、电荷和能量的粒子的引导中心轨道的统计损失。

**地址：**[itpplasma/SIMPLE：粒子损失估计的辛积分方法 --- itpplasma/SIMPLE: Symplectic Integration Methods for Particle Loss Estimation (github.com)](https://github.com/itpplasma/SIMPLE/tree/main)

该代码依赖库有：

- FORTRAN编译器，GNU Fortran或Intel Fortran
- 数据库NETCDF
- 计算库BLAS/LAPACK

首先通过conda获得依赖库

```
conda install compilers gfortran openmpi-mpifort openblas scalapack netcdf-fortran
conda install cmake 
```

之后构建

```shell
cd /path/to/SIMPLE
mkdir build; cd build
cmake ..
make
```

**可能存在的问题**

在集群上可能始终无法定位到某些软件包，尤其是无法正确定位NETCDFXX，这些标识可以在CMakeLists.txt中找到，如果始终缺失，可以在cmake构建时通过`-D`指定标识符的值，类似

```
cmake -DNETCDFINCLUDE_DIR=/home/dmcxe/miniconda3/envs/sims/include/ ..
```




# SPEC

[The Stepped Pressure Equilibrium Code (SPEC)](https://github.com/PrincetonUniversity/SPEC) 是一款基于 [MRxMHD Model](https://doi.org/10.1063/1.4765691) 的三维磁流体平衡求解程序。

这个文档中的安装教程和官方提供的[安装文档](https://princetonuniversity.github.io/SPEC/md_Compile.html)有所不同，在官方文档中使用`conda`来对安装环境和依赖包进行管理，在这个文档中直接通过Makefile安装SPEC。

### 源码下载
在终端输入
```
git clone https://github.com/PrincetonUniversity/SPEC.git
```

### 安装所依赖的工具和包

有两种方式进行，这里展示依赖apt的安装方法（gfortran_ubuntu）与依赖conda的安装方法（gfortran_conda）

#### Ubuntu

在终端依次输入

```shell
sudo apt install make
sudo apt install gfortran
sudo apt install libopenmpi-dev
sudo apt install liblapack-dev
sudo apt install m4
sudo apt install libfftw3-dev
sudo apt install libhdf5-dev
```

#### Conda

首先激活安装环境

```shell
conda activate <your env name>
```

在终端依次输入

```shell
conda install make
conda install gfortran
conda install openmpi
conda install liblapack
conda install m4
conda install fftw3
conda install hdf5
```

### 编译

#### Ubuntu

```
cd /path/to/SPEC
make BUILD_ENV=gfortran_ubuntu
```

详细配置可以看SPEC源文件中的`/path/to/SPEC/Makefile`和`/path/to/SPEC/SPECfile`两个文档

#### Conda

首先打开`/path/to/SPEC/SPECfile`下，在判断环境部分（例如109行之后）新增conda对应的编译指令。（这里是在M1 Macbook下编译，但Ubuntu上路径命名逻辑一致）。指令中`/Users/dmcxe/miniforge3/envs/simsopt/`需要修改为对应conda环境路径。

```makefile
ifeq ($(BUILD_ENV),gfortran_conda)
 # Build on M1 Mac with conda
 FC=mpif90
 FLAGS=-fPIC
 CFLAGS=-fdefault-real-8
 LINKS=-Wl,-rpath -Wl,/Users/dmcxe/miniforge3/envs/simsopt/lib/lapack -llapack -lblas
 LIBS=-I/Users/dmcxe/miniforge3/envs/simsopt/include
 LINKS+=-L/Users/dmcxe/miniforge3/envs/simsopt/lib -lhdf5_fortran -lhdf5 -lpthread -lz -lm
 LIBS+=-I/Users/dmcxe/miniforge3/envs/simsopt/include
 LINKS+=-lfftw3
 RFLAGS=-O2 -ffixed-line-length-none -ffree-line-length-none -fexternal-blas
 DFLAGS=-g -fbacktrace -fbounds-check -ffree-line-length-none -fexternal-blas -DDEBUG
endif
```

之后在目录下编译

```
cd /path/to/SPEC
make BUILD_ENV=gfortran_conda
```

详细配置可以看SPEC源文件中的`/path/to/SPEC/Makefile`和`/path/to/SPEC/SPECfile`两个文档

### 

### 将SPEC添加到环境变量
使用`vim ~/.bashrc`打开终端配置文件，在其中加入一行
```
export PATH=$PATH:~/Codes/SPEC
```
再在终端输入
```
source ~/.bashrc
```
如果一切顺利的话，此时你在终端输入`which xspec`，屏幕上将出现类似的下述输出
```
/home/plasma/Codes/xspec
```

### 测试
在终端依次输入下述命令
```
mkdir ~/SPEC_runs
cd ~/SPEC_runs
cp /path/to/SPEC/InputFiles/TestCases/G3V01L0Fi.001.sp .
xspec G3V01L0Fi.001.sp
```
这个时候将会开始执行SPEC的一个测试算例，等待一会儿，如果屏幕最后几行出现类似的输出
```
ending :       0.88 : myid=  0 ; completion ; time=      0.88s =     0.01m =   0.00h =  0.00d ; date= 2022/02/17 ; time= 17:35:33 ; ext = G1V02L0Fi.001                                               
ending :            : 
xspech :            :
xspech :       0.88 : myid=  0 : time=    0.01m =   0.00h =  0.00d ;
```
说明SPEC安装成功

### 后处理
SPEC的后处理工具分别基于`matlab`和`python`的，两者分别在`/path/to/SPEC/Utilities/matlabtools/`和`/path/to/SPEC/Utilities/pythontools/`文件夹下。
