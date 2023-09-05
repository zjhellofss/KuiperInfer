FROM --platform=linux/amd64 ubuntu:22.04

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 系统更新以及安装编译器等工具
RUN apt -y update
RUN apt -y upgrade
RUN apt install -y build-essential cmake git gfortran wget

# 编译安装OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git
RUN mkdir OpenBLAS/build
RUN cmake OpenBLAS/ -B OpenBLAS/build/ -DUSE_OPENMP=ON
RUN make -C OpenBLAS/build/ -j$(nproc)
RUN make install -C OpenBLAS/build/

# 下载armadillo的其他依赖
RUN apt install -y liblapack-dev libarpack2-dev libsuperlu-dev

# 编译安装armadillo
RUN wget https://sourceforge.net/projects/arma/files/armadillo-12.6.3.tar.xz
RUN tar -vxf armadillo-12.6.3.tar.xz
RUN mkdir armadillo-12.6.3/build
RUN cmake armadillo-12.6.3/ -B armadillo-12.6.3/build/
RUN cmake armadillo-12.6.3 -B armadillo-12.6.3/build/
RUN make -C armadillo-12.6.3/build -j$(nproc)
RUN make install -C armadillo-12.6.3/build

# 编译安装googletest
RUN git clone https://github.com/google/googletest.git
RUN mkdir googletest/build
RUN cmake googletest/ -B googletest/build/
RUN make -C googletest/build/ -j$(nproc)
RUN make install -C googletest/build/

# 编译安装googlebenchmark
RUN git clone https://github.com/google/benchmark.git
RUN mkdir benchmark/build
RUN cmake benchmark/ -B benchmark/build/ -DBENCHMARK_ENABLE_TESTING=OFF
RUN make -C benchmark/build/ -j$(nproc)
RUN make install -C benchmark/build/

# 编译安装googlelog
RUN git clone https://github.com/google/glog.git
RUN mkdir glog/build
RUN cmake glog/ -B glog/build/
RUN make -C glog/build/ -j$(nproc)
RUN make install -C glog/build/

RUN rm -rf OpenBLAS/ armadillo-12.6.3.tar.xz armadillo-12.6.3 googletest/ benchmark/ glog/

WORKDIR /app
