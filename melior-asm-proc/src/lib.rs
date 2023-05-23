#![feature(proc_macro_diagnostic, proc_macro_span_shrink)]
#![deny(warnings)]

use self::{input::MacroInput, verify::verify_mlir};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, LitStr};
use transform::transform_with_opt;

mod input;
mod transform;
mod verify;

#[proc_macro]
pub fn mlir_asm(input: TokenStream) -> TokenStream {
    // Parse macro input.
    let input = parse_macro_input!(input as MacroInput);

    // Verify MLIR code.
    verify_mlir(input.mlir_code.clone());

    // Transform using `mlir-opt`.
    let mlir_code = if let Some(opt_pass) = input.opt_pass {
        let opt_flags = opt_pass.flags.iter().map(LitStr::value).collect::<Vec<_>>();

        transform_with_opt(input.mlir_code, opt_flags.as_slice())
    } else {
        input.mlir_code.span().source_text().unwrap()
    };

    // Generate output.
    let context = input.context;
    quote! {
        melior_next::ir::Module::parse(#context, #mlir_code)
            .expect("Invalid MLIR code in mlir_asm!() proc macro.")
    }
    .into()
}
