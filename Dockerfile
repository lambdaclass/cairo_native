FROM debian:12

# Install dependencies
RUN apt update -y && apt install -y lsb-release \
    wget \
    curl \
    git \
    build-essential \
    libclang-dev \
    libz-dev

# Install LLVM 17
RUN echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-17 main" > /etc/apt/sources.list.d/llvm-17.list
RUN echo "deb-src http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-17 main" >> /etc/apt/sources.list.d/llvm-17.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt update -y && apt install -y llvm-17 \
    libmlir-17-dev \
    mlir-17-tools \
    libpolly-17-dev

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy cairo_native code
COPY . /cairo_native/

# Compile cairo_native
WORKDIR /cairo_native/
ENV MLIR_SYS_170_PREFIX=/usr/lib/llvm-17
ENV LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
ENV TABLEGEN_170_PREFIX=/usr/lib/llvm-17
RUN make deps
RUN make build
