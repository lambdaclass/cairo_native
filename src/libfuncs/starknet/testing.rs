use crate::error::Result;

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &str,
) -> Result<(), Error> {
    match selector {
        "set_sequencer_address" => {
            // self.starknet_state.exec_info.block_info.sequencer_address =
            //     as_single_input(inputs)?;
        }
        "set_block_number" => {
            // self.starknet_state.exec_info.block_info.block_number = as_single_input(inputs)?;
        }
        "set_block_timestamp" => {
            // self.starknet_state.exec_info.block_info.block_timestamp = as_single_input(inputs)?;
        }
        "set_caller_address" => {
            // self.starknet_state.exec_info.caller_address = as_single_input(inputs)?;
        }
        "set_contract_address" => {
            // self.starknet_state.exec_info.contract_address = as_single_input(inputs)?;
        }
        "set_version" => {
            // self.starknet_state.exec_info.tx_info.version = as_single_input(inputs)?;
        }
        "set_account_contract_address" => {
            // self.starknet_state.exec_info.tx_info.account_contract_address =
            // as_single_input(inputs)?;
        }
        "set_max_fee" => {
            // self.starknet_state.exec_info.tx_info.max_fee = as_single_input(inputs)?;
        }
        "set_transaction_hash" => {
            // self.starknet_state.exec_info.tx_info.transaction_hash = as_single_input(inputs)?;
        }
        "set_chain_id" => {
            // self.starknet_state.exec_info.tx_info.chain_id = as_single_input(inputs)?;
        }
        "set_nonce" => {
            // self.starknet_state.exec_info.tx_info.nonce = as_single_input(inputs)?;
        }
        "set_signature" => {
            // self.starknet_state.exec_info.tx_info.signature = inputs;
        }
        "pop_log" => {
            // let contract_logs = self.starknet_state.logs.get_mut(&as_single_input(inputs)?);
            // if let Some((keys, data)) =
            //     contract_logs.and_then(|contract_logs| contract_logs.events.pop_front())
            // {
            //     res_segment.write(keys.len())?;
            //     res_segment.write_data(keys.iter())?;
            //     res_segment.write(data.len())?;
            //     res_segment.write_data(data.iter())?;
            // }
        }
        "pop_l2_to_l1_message" => {
            // let contract_logs = self.starknet_state.logs.get_mut(&as_single_input(inputs)?);
            // if let Some((to_address, payload)) = contract_logs
            //     .and_then(|contract_logs| contract_logs.l2_to_l1_messages.pop_front())
            // {
            //     res_segment.write(to_address)?;
            //     res_segment.write(payload.len())?;
            //     res_segment.write_data(payload.iter())?;
            // }
        }
        _ => unimplemented!(), // Err(HintError::CustomHint(Box::from(format!("Unknown cheatcode selector: {selector}"))))?,
    }

    Ok(())
}
