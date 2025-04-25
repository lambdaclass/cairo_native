use bincode::de::read::SliceReader;
use sierra2casm_dbg::{decode_instruction, GraphMappings, Memory, Trace, ValueId};
use starknet_types_core::felt::Felt;
use std::{collections::HashMap, fs, str::FromStr};

fn main() {
    let memory = Memory::decode(SliceReader::new(&fs::read("memory-2.bin").unwrap()));
    let trace = Trace::decode(SliceReader::new(&fs::read("trace-2.bin").unwrap()));

    let mappings = GraphMappings::new(&memory, &trace, &HashMap::new());

    // let value_idx = memory
    //     .iter()
    //     .copied()
    //     .enumerate()
    //     .filter_map(|(idx, val)| {
    //         (val == Some(Felt::from_str("9980669810").unwrap())).then_some(idx)
    //     })
    //     .collect::<Vec<_>>();

    let value_idx = ValueId(93139);

    println!("Memory offset: {value_idx:?}");
    for id in &mappings[value_idx] {
        println!(
            "{id:?} => {} [{:?}]",
            decode_instruction(&memory, trace[id.0].pc),
            trace[id.0]
        );
    }

    println!("[93139] = {}", memory[93139].unwrap());
    println!("[93140] = {}", memory[93140].unwrap());
    println!("[93142] = {}", memory[93142].unwrap());

    // dbg!(memory[25386].unwrap().to_string());
    // dbg!(value_idx);
}
