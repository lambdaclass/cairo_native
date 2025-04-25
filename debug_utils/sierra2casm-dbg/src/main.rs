use bincode::de::read::SliceReader;
use cairo_vm::serde::deserialize_program::HintParams;
use clap::Parser;
use serde_json::Value;
use sierra2casm_dbg::{
    decode_instruction, run_search_algorithm,
    search::{DfsQueue, NodeId},
    GraphMappings, Memory, Trace,
};
use starknet_types_core::felt::Felt;
use std::{collections::HashMap, fs, path::PathBuf, str::FromStr};

#[derive(Debug, Parser)]
struct CmdArgs {
    #[clap(long)]
    memory_path: PathBuf,
    #[clap(long)]
    trace_path: PathBuf,
    #[clap(long)]
    program_path: Option<PathBuf>,

    #[clap(short, long, value_parser = parse_felt252)]
    source_value: Felt,
    #[clap(short, long, value_parser = parse_felt252)]
    target_value: Felt,
}

fn parse_felt252(input: &str) -> Result<Felt, String> {
    Felt::from_str(input).map_err(|e| e.to_string())
}

fn main() {
    let args = CmdArgs::parse();

    //
    // Load data from disk.
    //
    println!("Loading memory and trace.");
    let memory = Memory::decode(SliceReader::new(&fs::read(args.memory_path).unwrap()));
    let trace = Trace::decode(SliceReader::new(&fs::read(args.trace_path).unwrap()));
    println!("  {:?}", trace.first().unwrap());
    println!("  {:?}", trace.last().unwrap());

    let hints = match args.program_path {
        None => HashMap::default(),
        Some(program_path) => {
            println!("Loading hints from provided program JSON.");
            let mut program_json: Value =
                serde_json::from_slice(&fs::read(program_path).unwrap()).unwrap();
            let hints = program_json
                .as_object_mut()
                .unwrap()
                .remove("hints")
                .unwrap();

            let hints: HashMap<usize, Vec<HintParams>> = serde_json::from_value(hints).unwrap();
            hints
                .into_iter()
                .map(|(key, value)| {
                    //
                    (
                        key,
                        value
                            .into_iter()
                            .map(|hint_params| serde_json::from_str(&hint_params.code).unwrap())
                            .collect(),
                    )
                })
                .collect()
        }
    };

    //
    // Generate graph mappings.
    //
    println!("Generating graph mappings.");
    let mappings = GraphMappings::new(&memory, &trace, &hints);

    //
    // Find initial and final values.
    //
    println!("Finding initial and final values within the data.");
    let source_value = mappings
        .value2step()
        .keys()
        .copied()
        .filter(|x| memory[x.0] == Some(args.source_value))
        .min()
        .expect("Source value not found within accessed memory.");
    let target_value = mappings
        .value2step()
        .keys()
        .copied()
        .filter(|x| memory[x.0] == Some(args.target_value))
        .max()
        .expect("Target value not found within accessed memory.");
    println!("  Source value found at {}.", source_value.0);
    println!("  Target value found at {}.", target_value.0);

    println!();

    //
    // Find a path between the source and target nodes.
    //
    // Queue containers:
    //   - BfsQueue: Will find the shortest path using the BFS algorithm.
    //   - DfsQueue: Will find the left-most path using the DFS algorithm.
    //
    println!("Starting search algorithm.");
    let mut iter =
        run_search_algorithm::<DfsQueue<_>>(&memory, &mappings, source_value, target_value);
    println!();
    println!();

    let mut num_solutions = 0;
    while let Some(path) = iter.next() {
        num_solutions += 1;

        println!("Found solution at step {}.", iter.queue().current_step());
        // println!("Connecting path (spans {} steps):", path.len() >> 1);
        // for id in path {
        //     match id {
        //         NodeId::Step(offset) => {
        //             if let Some(hints) = hints.get(&offset.0) {
        //                 for hint in hints {
        //                     println!("; {}", hint.representing_string());
        //                 }
        //             }
        //             println!("{}", decode_instruction(&memory, trace[offset.0].pc));
        //             println!("    {:?}", trace[offset.0]);
        //         }
        //         NodeId::Value(offset) => {
        //             println!("  [{}] = {}", offset.0, memory[offset.0].unwrap());
        //             println!();
        //         }
        //     }
        // }
        // println!();

        let mut prev_value: Option<Felt> = None;
        for id in path.into_iter().filter_map(|x| match x {
            NodeId::Step(_) => None,
            NodeId::Value(id) => Some(id),
        }) {
            let curr_value = memory[id.0].unwrap();
            match prev_value {
                Some(prev_value) if curr_value != prev_value => println!(
                    "  [{}] = {curr_value} (Δ{})",
                    id.0,
                    curr_value.to_bigint() - prev_value.to_bigint()
                ),
                None => println!("  [{}] = {curr_value}", id.0),
                _ => {}
            }
            prev_value = Some(curr_value);
        }
        println!();
        println!();
    }

    println!("Done! Found {num_solutions} solutions.");
}
