use cairo_lang_starknet_classes::contract_class::ContractClass;
use cairo_native::context::NativeContext;
use test_case::test_case;

#[test_case("tests/tests/starknet/contracts/kakarot/contracts_AccountContract.contract_class.json")]
#[test_case("tests/tests/starknet/contracts/kakarot/contracts_Cairo1Helpers.contract_class.json")]
#[test_case(
    "tests/tests/starknet/contracts/kakarot/contracts_Cairo1HelpersFixture.contract_class.json"
)]
#[test_case("tests/tests/starknet/contracts/kakarot/contracts_KakarotCore.contract_class.json" => ignore["failing function_call compilation with index out of bound"])]
#[test_case(
    "tests/tests/starknet/contracts/kakarot/contracts_UninitializedAccount.contract_class.json"
)]
fn compile_to_native(file_path: &str) {
    let native_context = NativeContext::default();

    let contract_class: ContractClass =
        serde_json::from_str(&std::fs::read_to_string(file_path).expect("failed to read file"))
            .expect("failed to parse json");
    let program = contract_class
        .extract_sierra_program()
        .expect("failed to extract program");

    native_context
        .compile(&program, None)
        .expect("failed to compile");
}
