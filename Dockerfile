FROM debian:12

# Install dependencies
RUN apt update -y && apt install -y lsb-release \
    wget \
    curl \
    git \
    build-essential \
    libclang-dev \
    libz-dev \
    libzstd-dev

# Install LLVM 20
RUN echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" > /etc/apt/sources.list.d/llvm-20.list
RUN echo "deb-src http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-20 main" >> /etc/apt/sources.list.d/llvm-20.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt update -y && apt install -y \
    libmlir-20-dev \
    libpolly-20-dev \
    llvm-20-dev \
    mlir-20-tools

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy cairo_native code
COPY . /cairo_native/

# Compile cairo_native
WORKDIR /cairo_native/
ENV MLIR_SYS_200_PREFIX=/usr/lib/llvm-20
ENV LLVM_SYS_201_PREFIX=/usr/lib/llvm-20
ENV TABLEGEN_200_PREFIX=/usr/lib/llvm-20
RUN make deps
RUN make build
