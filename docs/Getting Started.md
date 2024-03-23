# Introduction to Cairo

Cairo is a statically typed language and also a provable language. In other words, you must declare the type of the variable.

When you declare Cairo, it then compile into SIERRA and the CASM in the Cairo Virtual Machine(CVM).
Tooling.

It is important to make sure this three items are working in your PC

    Scarb — package manager and compiles smart contract.
    Katana — local development environment(serves as our sequencer)
    Starkli — CLI tool for declaration, deployment, interacting. The interactions can be calls and invocations.

# Installation

# Scarb: The Package Manager

To make the most of this chapter, a basic grasp of the Cairo programming
language is advised. We suggest reading chapters 1-6 of the [Cairo
Book](https://book.cairo-lang.org/title-page.html), covering topics from
_Getting Started_ to _Enums and Pattern Matching._ Follow this by
studying the [Starknet Smart Contracts
chapter](https://book.cairo-lang.org/ch12-00-introduction-to-starknet-smart-contracts.html)
in the same book. With this background, you’ll be well-equipped to
understand the examples presented here.

Scarb is Cairo’s package manager designed for both Cairo and Starknet
projects. It handles dependencies, compiles projects, and integrates
with tools like Foundry. It is built by the same team that created
Foundry for Starknet.

# Scarb Workflow

Follow these steps to develop a Starknet contract using Scarb:

1.  **Initialize:** Use `scarb new` to set up a new project, generating
    a `Scarb.toml` file and initial `src/lib.cairo`.

2.  **Code:** Add your Cairo code in the `src` directory.

3.  **Dependencies:** Add external libraries using `scarb add`.

4.  **Compile:** Execute `scarb build` to convert your contract into
    Sierra code.

Scarb simplifies your development workflow, making it efficient and
streamlined.

# Installation

Cairo can be installed by simply downloading [Scarb](https://docs.swmansion.com/scarb/docs). Scarb bundles the Cairo compiler and the Cairo language server together in an easy-to-install package so that you can start writing Cairo code right away.

Scarb is also Cairo's package manager and is heavily inspired by [Cargo](https://doc.rust-lang.org/cargo/), Rust’s build system and package manager.

Scarb handles a lot of tasks for you, such as building your code (either pure Cairo or Starknet contracts), downloading the libraries your code depends on, building those libraries, and provides LSP support for the VSCode Cairo 1 extension.

As you write more complex Cairo programs, you might add dependencies, and if you start a project using Scarb, managing external code and dependencies will be a lot easier to do.

Let's start by installing Scarb.

## Installing Scarb

### Requirements

To install Scarb, please refer to the [installation instructions](https://docs.swmansion.com/scarb/download) We strongly recommend that you install Scarb via [asdf](https://docs.swmansion.com/scarb/download.html#install-via-asdf), a CLI tool that can manage multiple language runtime versions on a per-project basis. This will ensure that the version of Scarb you use to work on a project always matches the one defined in the project settings, avoiding problems related to version mismatches.

Please refer to the [asdf documentation](https://asdf-vm.com/guide/getting-started.html) to install all prerequisites.

Once you have asdf installed locally, you can download Scarb plugin with the following command:

```
asdf plugin add scarb

```
and set a global version:
```
asdf global scarb 2.6.3

```
Otherwise, you can simply run the following command in your terminal, and follow the onscreen instructions. This will install the latest stable release of Scarb.

```
curl --proto '=https' --tlsv1.2 -sSf https://docs.swmansion.com/scarb/install.sh | sh

```
In both cases, you can verify installation by running the following command in a new terminal session, it should print both Scarb and Cairo language versions, e.g: 

```
$ scarb --version
scarb 2.6.3 (e6f921dfd 2024-03-13)
cairo: 2.6.3 (https://crates.io/crates/cairo-lang-compiler/2.6.3)
sierra: 1.5.0

```

Scarb requires a Git executable to be available in the PATH environment variable.

# Cairo Project Structure

Next, we’ll dive into the key components that make up a Cairo project.

## Cairo Packages

Cairo packages, also referred to as "crates" in some contexts, are the
building blocks of a Cairo project. Each package must follow several
rules:

- A package must include a `Scarb.toml` file, which is Scarb’s
  manifest file. It contains the dependencies for your package.

- A package must include a `src/lib.cairo` file, which is the root of
  the package tree. It allows you to define functions and declare used
  modules.

Package structures might look like the following case where we have a
package named `my_package`, which includes a `src` directory with the
`lib.cairo` file inside, a `snips` directory which in itself a package
we can use, and a `Scarb.toml` file in the top-level directory.

    my_package/
    ├── src/
    │   ├── module1.cairo
    │   ├── module2.cairo
    │   └── lib.cairo
    ├── snips/
    │   ├── src/
    │   │   ├── lib.cairo
    │   ├── Scarb.toml
    └── Scarb.toml

Within the `Scarb.toml` file, you might have:

    [package]
    name = "my_package"
    version = "0.1.0"

    [dependencies]
    starknet = ">=2.0.1"
    

Here starknet are the dependencies of the package. The
`starknet` dependency is hosted on the Scarb registry .


# Setting Up a Project with Scarb

To create a new project using Scarb, navigate to your desired project
directory and execute the following command:

    $ scarb new hello_scarb

This command will create a new project directory named `hello_scarb`,
including a `Scarb.toml` file, a `src` directory with a `lib.cairo` file
inside, and initialize a new Git repository with a `.gitignore` file.

    hello_scarb/
    ├── src/
    │   └── lib.cairo
    └── Scarb.toml

Upon opening `Scarb.toml` in a text editor, you should see something
similar to the code snippet below:

    [package]
    name = "hello_scarb"
    version = "0.1.0"

    # See more keys and their definitions at https://docs.swmansion.com/scarb/docs/reference/manifest.html
    [dependencies]
    # foo = { path = "vendor/foo" }

# Building a Scarb Project

Cairo can be used for building programs such as Libraries for Arithmetic operations . Cairo programs can not access the state, only a starknet contract can access the starknet state. When declaring a program, you can start with the key word below:

```
fn main()
```
Let’s code a program

```
use debug::PrintTrait;

mod Sum_numbers;
mod Calculations;

fn main() {
    GM_CAIRO();
    into_to_felt();
    let result = Calculations::add(30, 10);
    let result2 = Calculations::subtract(20, 10);
    let result3 = Calculations::mul(30, 10);
    let result4 = Calculations::div(20, 5);
}

fn GM_CAIRO() {
    'GM_CAIRO'.print();
}

fn into_to_felt() {
    let felt_1 = 'chris';
    felt_1.print();

    let felt_2 = 'true';
    felt_2.print();

    let felt_3 = '30';
    felt_3.print();
}

#[cfg(test)]
mod tests {
    // use super::fib;

    #[test]
    #[available_gas(100000)]
    fn it_works() { // assert(fib(16) == 987, 'it works!');
    }
}


```

This code is modularization for a neat job.

```
use debug::PrintTrait;

fn add(num_1: u32, num_2: u32) -> u32 {
    let result: u32 = num_1 + num_2;
    result.print();
    return result;
}

fn subtract(num_1: u32, num_2: u32) -> u32 {
    let result2: u32 = num_1 - num_2;
    result2.print();
    result2
}
fn mul(num_1: u32, num_2: u32) -> u32 {
    let result3: u32 = num_1 * num_2;
    result3.print();
    result3
}
fn div(num_1: u32, num_2: u32) -> u32 {
    let result4: u32 = num_1 / num_2;
    result4.print();
    result4
}
fn check_sum(num_1: usize, num_2: usize) -> bool {
    let result5: usize = add(num_1, num_2);

    if result5 % 2 == 0 {
        'true'.print();
        true
    } else {
        'false'.print();
        false
    }
}
```
Let print our result using this command,
```
Run scarb cairo -run
```

To build (compile) your project from your directory, use
the following command:

    scarb build



