# MLIR Resources

## How MLIR Works
MLIR is composed of **dialects**, which is like a IR of it's own, and this
IR can be converted to another dialect IR (if the functionality exists).
This is what makes MLIR shine.

Some commonly used dialects in this project:
- The arith dialect: It contains arithmetic operations, such as `addi`, `subi` for addition and subtraction.
- The cf dialect: It contains basic control flow operations, such as the `br` and `cond_br`, which are unconditional and conditional jumps.

### The IR
The MLIR IR is composed recursively like this: `Operation -> Region -> Block -> Operations`

Each operation has 1 or more region, each region has 1 or more blocks, each
block has 1 or more operations.

This way a MLIR program can be composed.

### Transformations and passes
MLIR provides a set of transformations that can optimize the IR.
Such as `canonicalize`.

Check out <https://mlir.llvm.org/docs/Canonicalization/> and <https://mlir.llvm.org/docs/Passes/>.

### Translating
In our case, llvm is our target, so we end up translating all dialects down
to the LLVM dialect, which then gets converted to LLVM IR.

## Learning Resources
Resources marked with **→** are best.

- Introduction
    - **→** [2019 EuroLLVM Developers’ Meeting: MLIR: Multi-Level Intermediate Representation Compiler Infrastructure](https://www.youtube.com/watch?v=qzljG6DKgic)
    - → [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/pdf/2002.11054.pdf)
    The paper introducing the MLIR framework
        - 7-minute video summary of paper:
        [Read a paper: Multi-level Intermediate Representation (MLIR)](https://www.youtube.com/watch?v=6BwqK6E8v3g)
        - Another version of the paper:
        [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/85bf23fe88bd5c7ff60365bd0c6882928562cbeb.pdf)
- MLIR Tutorial
    - **→** (slides) [MLIR Tutorial (LLVM Dev Mtg, 2020)](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf)
    - **→** (video) [2020 LLVM Developers’ Meeting: M. Amini & R. Riddle “MLIR Tutorial”](https://www.youtube.com/watch?v=Y4SvqTtOIDk)
    - (older slides) [MLIR Tutorial (LLVM Developers Meeting, Euro-LLVM 2019)](https://llvm.org/devmtg/2019-04/slides/Tutorial-AminiVasilacheZinenko-MLIR.pdf)
    - (older slides) [MLIR Tutorial (MLIR 4 HPC, 2019)](https://users.cs.utah.edu/~mhall/mlir4hpc/pienaar-MLIR-Tutorial.pdf)
    - (older video) [2019 EuroLLVM Developers’ Meeting: Mehdi & Vasilache & Zinenko “Building a Compiler with MLIR”](https://www.youtube.com/watch?v=cyICUIZ56wQ)
- **→** Another MLIR Tutorial
[https://github.com/j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial)
- **→** [How to build a compiler with LLVM and MLIR](https://www.youtube.com/playlist?list=PLlONLmJCfHTo9WYfsoQvwjsa5ZB6hjOG5)
- Other articles, posts
    - **→** [Intro to LLVM and MLIR with Rust and Melior](https://edgarluque.com/blog/mlir-with-rust/)
    - **→** [MLIR Notes](http://lastweek.io/notes/MLIR/)
    - **→** [Compilers and IRs: LLVM IR, SPIR-V, and MLIR](https://www.lei.chat/posts/compilers-and-irs-llvm-ir-spirv-and-mlir/) [[HN]](https://news.ycombinator.com/item?id=33387149)
    - [MLIR: Redefining the compiler infrastructure](https://iq.opengenus.org/mlir-compiler-infrastructure/)
    - [Pinch: Implementing a borrow-checked language with MLIR](https://badland.io/pinch.md)
- [Official Documentation](https://mlir.llvm.org/docs/)
    - [MLIR Homepage](https://mlir.llvm.org/)
    - [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
    - [MLIR Compiler](https://www.youtube.com/MLIRCompiler) Youtube Channel

### Talks, Presentations, & Videos
- [2020 LLVM in HPC Workshop: Keynote: MLIR: an Agile Infrastructure for Building a Compiler Ecosystem](https://www.youtube.com/watch?v=0bxyZDGs-aA)
- [2021 LLVM Dev Mtg “Representing Concurrency with Graph Regions in MLIR”](https://www.youtube.com/watch?v=Vfk9n3ir_5s)
- [2022 LLVM Dev Mtg: Paths towards unifying LLVM and MLIR](https://www.youtube.com/watch?v=VbFqA9rvxPs)
- [2022 LLVM Dev Mtg: VAST: MLIR for program analysis of C/C++](https://www.youtube.com/watch?v=YFqWa4pxXzM)
- [2022 LLVM Dev Mtg: MLIR for Functional Programming](https://www.youtube.com/watch?v=cyMQbZ0B84Q)
- [2022 EuroLLVM Dev Mtg “Prototyping a Compiler for Homomorphic Encryption Using MLIR”](https://www.youtube.com/watch?v=QyxiqmO6_qQ)
- [cirgen: MLIR based compiler for zk-STARK circuit generation - Frank Laub (RISC Zero)](https://www.youtube.com/watch?v=TsP14-hI_W0)
- [Prototyping a compiler for homomorphic encryption using MLIR](https://www.youtube.com/watch?v=F9qXBuSkQFY)
    - [Slides](https://llvm.org/devmtg/2022-04-03/slides/Prototyping.a.compiler.for.homomorphic.encryption.in.MLIR.pdf)
- [The HEIR Compiler w/ Jeremy Kun](https://www.youtube.com/watch?v=ne5D_kqlxYg)

### Useful code
- [`raviqqe/melior`](https://github.com/raviqqe/melior)
- [`femtomc/mlir-sys`](https://github.com/femtomc/mlir-sys) Rust bindings to the MLIR C API.
- [`GetFirefly/firefly`](https://github.com/GetFirefly/firefly) An alternative BEAM implementation, designed for WebAssembly.
