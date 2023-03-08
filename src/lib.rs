//! A compiler to convert Cairo's intermediate representation "Sierra" code to MLIR.

//#![deny(warnings)]
#![warn(clippy::nursery)]
#![allow(unused)]

use std::{fs, path::PathBuf, time::Instant};

use cairo_lang_sierra::{program::Program, ProgramParser};
use clap::Parser;
use melior_next::{
    dialect,
    ir::{operation, Attribute, Block, Identifier, Location, Module, Region, Type, Value},
    pass,
    utility::{register_all_dialects, register_all_passes},
    Context, ExecutionEngine,
};

use crate::compiler::Compiler;

pub mod compiler;
mod libfuncs;
mod statements;
mod types;
