FROM debian:12

# Install dependencies
RUN apt update -y && apt install -y lsb-release \
    wget \
    curl \
    git \
    build-essential \
    libclang-dev \
    libz-dev

# Install LLVM 16
RUN echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-16 main" > /etc/apt/sources.list.d/llvm-16.list
RUN echo "deb-src http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-16 main" >> /etc/apt/sources.list.d/llvm-16.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt update -y && apt install -y llvm-16 \
    libmlir-16-dev \
    mlir-16-tools \
    libpolly-16-dev

# Install rust nightly-2023-06-19 (1.72.0-nightly)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y --default-toolchain=nightly-2023-07-29
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy cairo_native code
COPY . /cairo_native/

# Fetch cairo corelibs
RUN git clone --depth 1 \
    --branch v2.1.0-rc3 \
    https://github.com/starkware-libs/cairo.git \
    starkware-cairo
RUN cp -r starkware-cairo/corelib /cairo_native
RUN rm -rf starkware-cairo/

# Compile cairo_native
WORKDIR /cairo_native/
ENV MLIR_SYS_160_PREFIX=/usr/lib/llvm-16
RUN cargo +nightly-2023-07-29 build --release --all-features --locked
