use clap::Parser;
use sierra_emu::EntryPoint;
use std::path::PathBuf;

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
