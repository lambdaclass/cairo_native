use clap::Parser;
use std::{num::ParseIntError, path::PathBuf, str::FromStr};

#[derive(Debug, Parser)]
pub struct CmdArgs {
    pub program: PathBuf,
    pub entry_point: EntryPoint,

    pub args: Vec<String>,
    #[clap(long)]
    pub available_gas: Option<u64>,

    #[clap(long, short)]
    pub output: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub enum EntryPoint {
    Number(u64),
    String(String),
}

impl FromStr for EntryPoint {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.chars().next() {
            Some(x) if x.is_numeric() => Self::Number(s.parse()?),
            _ => Self::String(s.to_string()),
        })
    }
}
