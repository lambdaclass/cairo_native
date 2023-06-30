# Multi-Level Intermediate Representation


## Useful notes

### LLVM alloca notes

It's better to use opaque pointers for the returned alloca data instead of specifying the full pointer type. Also LLVM is switching to those and deprecating non-opaque pointers.
### Creating a constant llvm array

To create a constant array, you need to use the `llvm.mlir.constant` operation.
The way to represent the array data is by using a dense element attribute, for example:

```mlir
// The attribute should look like this:
dense<[3, 1, 4, 1, 5]> : tensor<5 x i8>

// Full example
%alloca_size = llvm.mlir.constant(5 : i64) : i64
%array_ptr = llvm.alloca %alloca_size x i8 : (i64) -> !llvm.ptr
%array_data = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
llvm.store %array_data, %array_ptr : !llvm.array<5 x i8>, !llvm.ptr
```

## Online Resources

### Websites
- [MLIR Homepage](https://mlir.llvm.org/)
- [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/pdf/2002.11054.pdf)
- [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/85bf23fe88bd5c7ff60365bd0c6882928562cbeb.pdf)
- MLIR Tutorial
  - [MLIR Tutorial](https://llvm.org/devmtg/2019-04/slides/Tutorial-AminiVasilacheZinenko-MLIR.pdf)
  - [MLIR Tutorial](https://users.cs.utah.edu/~mhall/mlir4hpc/pienaar-MLIR-Tutorial.pdf)
  - [MLIR Tutorial](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Docs: Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)
- [MLIR Notes](http://lastweek.io/notes/MLIR/)
- [Compilers and IRs: LLVM IR, SPIR-V, and MLIR](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/) [[HN]](https://news.ycombinator.com/item?id=33387149)
- [MLIR: Redefining the compiler infrastructure](https://iq.opengenus.org/mlir-compiler-infrastructure/)
- [Pinch: Implementing a borrow-checked language with MLIR](https://badland.io/pinch.md)
- [Tensorflow: MLIR](https://www.tensorflow.org/mlir)

#### CUDA & GPU
- [Wikipedia: PTX](https://en.wikipedia.org/wiki/Parallel_Thread_Execution)
- [NVidia CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [User Guide for NVPTX Back-end](https://llvm.org/docs/NVPTXUsage.html)
  - [`nvgpu` dialect documentation](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [LLVM GPU code with NVPTX](https://wiki.aalto.fi/display/t1065450/LLVM+GPU+code+with+NVPTX)
- [Stackoverflow: CUDA: compilation of LLVM IR using NVPTX](https://stackoverflow.com/questions/23873113/cuda-compilation-of-llvm-ir-using-nvptx)
- [Rust CUDA Working Group](https://github.com/rust-cuda/wg)
- [GPU code generation status: NVidia, OpenCL](https://discourse.llvm.org/t/gpu-code-generation-status-nvidia-opencl/2080/1)
- [Lowering GPU dialect](https://discourse.llvm.org/t/lowering-gpu-dialect/3609)

#### General LLVM-related
- [Writing an LLVM backend for the Move language in Rust](https://brson.github.io/2023/03/12/move-on-llvm)

### Video
- [MLIR Compiler](https://www.youtube.com/MLIRCompiler)
- [Read a paper: Multi-level Intermediate Representation (MLIR)](https://www.youtube.com/watch?v=6BwqK6E8v3g)
- [2019 EuroLLVM Developers’ Meeting: T. Shpeisman & C. Lattner “MLIR: Multi-Level Intermediate Repr..”](https://www.youtube.com/watch?v=qzljG6DKgic)
- [2019 EuroLLVM Developers’ Meeting: Mehdi & Vasilache & Zinenko “Building a Compiler with MLIR”](https://www.youtube.com/watch?v=cyICUIZ56wQ)
- [2020 LLVM Developers’ Meeting: M. Amini & R. Riddle “MLIR Tutorial”](https://www.youtube.com/watch?v=Y4SvqTtOIDk)
- [2020 LLVM in HPC Workshop: Keynote: MLIR: an Agile Infrastructure for Building a Compiler Ecosystem](https://www.youtube.com/watch?v=0bxyZDGs-aA)
- [2021 LLVM Dev Mtg “Representing Concurrency with Graph Regions in MLIR”](https://www.youtube.com/watch?v=Vfk9n3ir_5s)
- [2022 LLVM Dev Mtg: Paths towards unifying LLVM and MLIR](https://www.youtube.com/watch?v=VbFqA9rvxPs)
- [2022 LLVM Dev Mtg: VAST: MLIR for program analysis of C/C++](https://www.youtube.com/watch?v=YFqWa4pxXzM)
- [2022 LLVM Dev Mtg: MLIR for Functional Programming](https://www.youtube.com/watch?v=cyMQbZ0B84Q)
- [2022 EuroLLVM Dev Mtg “Prototyping a Compiler for Homomorphic Encryption Using MLIR”](https://www.youtube.com/watch?v=QyxiqmO6_qQ)
- [MLIR-based code generation for GPU tensor cores](https://www.youtube.com/watch?v=3LLzHKeL2hs)
- [cirgen: MLIR based compiler for zk-STARK circuit generation - Frank Laub (RISC Zero)](https://www.youtube.com/watch?v=TsP14-hI_W0)
- [Prototyping a compiler for homomorphic encryption using MLIR](https://www.youtube.com/watch?v=F9qXBuSkQFY)
  - [Slides](https://llvm.org/devmtg/2022-04-03/slides/Prototyping.a.compiler.for.homomorphic.encryption.in.MLIR.pdf)
 
### Useful code
- [`femtomc/mlir-sys`](https://github.com/femtomc/mlir-sys) Rust bindings to the MLIR C API.
- [`raviqqe/melior`](https://github.com/raviqqe/melior)
  - [Issue #24 Roadmap to v1 for LLVM 15](https://github.com/raviqqe/melior/issues/24)
- [`edg-l/melior-next`](https://github.com/edg-l/melior-next) 
- [`lambdaclass/llvm-mlir-sys`](https://github.com/lambdaclass/llvm-mlir-sys)
- [`openxla/iree`](https://github.com/openxla/iree) A retargetable MLIR-based machine learning compiler and runtime toolkit.
- [`GetFirefly/firefly`](https://github.com/GetFirefly/firefly) An alternative BEAM implementation, designed for WebAssembly.
- [`zero9178/Pylir`](https://github.com/zero9178/Pylir) An optimizing ahead-of-time Python Compiler.
  - [Documentation](https://zero9178.github.io/Pylir/)
- [`plaidml/plaidml`](https://github.com/plaidml/plaidml) PlaidML is a framework for making deep learning work everywhere.
- [`yn224/mlir-gpu-playground`](https://github.com/yn224/mlir-gpu-playground)
- [`mmperf/mmperf`](https://github.com/mmperf/mmperf) MatMul Performance Benchmarks for a Single CPU Core comparing both hand engineered and codegen kernels.

### Background
- [Computation graphs and graph computation](https://breandan.net/2020/06/30/graph-computation/)
- [High-performance analytics: Why differential dataflow is the next level of query optimisation](https://tably.substack.com/p/high-performance-analytics)
- [`TimelyDataflow/differential-dataflow`](https://github.com/TimelyDataflow/differential-dataflow/)
- [Materialize: The Streaming Database You Already Know How to Use](https://materialize.com/)
- [Chris Lattner: Compilers, LLVM, Swift, TPU, and ML Accelerators | Lex Fridman Podcast #21](https://www.youtube.com/watch?v=yCd3CzGSte8)
- [Jim Keller: Moore's Law, Microprocessors, and First Principles | Lex Fridman Podcast #70](https://www.youtube.com/watch?v=Nb2tebYAaOA)
- [Jim Keller: The Future of Computing, AI, Life, and Consciousness | Lex Fridman Podcast #162](https://www.youtube.com/watch?v=G4hL5Om4IJ4&t=2990s)
- [Building the Software 2 0 Stack (Andrej Karpathy)](https://www.youtube.com/watch?v=y57wwucbXR8)
