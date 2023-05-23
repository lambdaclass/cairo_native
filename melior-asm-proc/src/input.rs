use proc_macro2::TokenStream;
use syn::{
    custom_keyword, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token::Paren,
    Expr, LitStr, Result, Token,
};

custom_keyword!(opt);

pub struct MacroInput {
    /// The MLIR context to use.
    pub context: Expr,
    pub opt_pass: Option<OptFlags>,
    /// AST `->` token.
    pub colon: Token![=>],
    /// The MLIR source code.
    pub mlir_code: TokenStream,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            context: input.parse()?,
            opt_pass: OptFlags::parse(input).ok(),
            colon: input.parse()?,
            mlir_code: {
                let mlir_code = input.cursor().token_stream();

                // Skip everything until EOF.
                input.step(|cursor| {
                    let mut cursor = *cursor;
                    while let Some((_, next_cursor)) = cursor.token_tree() {
                        cursor = next_cursor;
                    }
                    Ok(((), cursor))
                })?;

                mlir_code
            },
        })
    }
}

pub struct OptFlags {
    pub opt_kw: opt,
    pub paren: Paren,
    pub flags: Punctuated<LitStr, Token![,]>,
}

impl Parse for OptFlags {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(Self {
            opt_kw: input.parse()?,
            paren: parenthesized!(content in input),
            flags: content.parse_terminated(<LitStr as Parse>::parse, Token![,]).unwrap(),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quote::quote;

    #[test]
    fn parse_macro_input() {
        let input: MacroInput = syn::parse2(quote! { &ctx =>
            func.func @main() -> i32 {
                %0 = arith.constant 0 : i32
                return %0 : i32
            }
        })
        .unwrap();

        assert_eq!(input.context, syn::parse2(quote!(&ctx)).unwrap());
        assert_eq!(
            input.mlir_code.to_string(),
            quote! {
                func.func @main() -> i32 {
                    %0 = arith.constant 0 : i32
                    return %0 : i32
                }
            }
            .to_string()
        );
    }

    #[test]
    fn parse_opt_flags() {
        let input: OptFlags = syn::parse2(quote! {
            opt("--expand-strided-metadata")
        })
        .unwrap();

        dbg!(input.flags);
    }
}
