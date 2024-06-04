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

# Install LLVM 18
RUN echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" > /etc/apt/sources.list.d/llvm-18.list
RUN echo "deb-src http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" >> /etc/apt/sources.list.d/llvm-18.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt update -y && apt install -y llvm-18 \
    libmlir-18-dev \
    mlir-18-tools \
    libpolly-18-dev

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy cairo_native code
COPY . /cairo_native/

# Compile cairo_native
WORKDIR /cairo_native/
ENV MLIR_SYS_180_PREFIX=/usr/lib/llvm-18
ENV LLVM_SYS_180_PREFIX=/usr/lib/llvm-18
ENV TABLEGEN_180_PREFIX=/usr/lib/llvm-18
RUN make deps
RUN make build
