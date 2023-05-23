use melior_next::{
    dialect::Registry, ir::Module, utility::register_all_dialects, Context, LogicalResult,
};
use proc_macro2::{LineColumn, Span, TokenStream, TokenTree};
use syn::spanned::Spanned;

pub fn verify_mlir(mlir_stream: TokenStream) {
    let context = Context::new();

    let registry = Registry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);

    context.attach_diagnostic_handler(|diagnostic| {
        let location = {
            let loc = diagnostic.location().to_string();

            let (line, column) = loc
                .strip_prefix("loc(\"-\":")
                .unwrap()
                .strip_suffix(')')
                .unwrap()
                .split_once(':')
                .unwrap();

            LineColumn {
                line: line.parse().unwrap(),
                column: column.parse().unwrap(),
            }
        };
        let location_span = find_span_from_stream_and_location(mlir_stream.clone(), location);

        location_span.unwrap().before().error(diagnostic.to_string()).emit();
        LogicalResult::success()
    });

    Module::parse(&context, &mlir_stream.span().source_text().unwrap());
}

fn find_span_from_stream_and_location(stream: TokenStream, location: LineColumn) -> Span {
    let location = {
        let mut new_location = stream.span().start();

        new_location.line += location.line - 1;
        match location.line {
            1 => new_location.column += location.column,
            _ => new_location.column = location.column,
        }

        new_location
    };

    fn inner(stream: TokenStream, location: LineColumn) -> Option<Span> {
        for token_tree in stream.into_iter() {
            match token_tree {
                TokenTree::Group(group) => {
                    if group.span_open().start() == location {
                        return Some(group.span_open());
                    }

                    if let Some(x) = inner(group.stream(), location) {
                        return Some(x);
                    }

                    if group.span_close().start() == location {
                        return Some(group.span_close());
                    }
                }
                _ => {
                    if token_tree.span().start() == location {
                        return Some(token_tree.span());
                    }
                }
            }
        }

        None
    }

    let fallback_span = stream.span();
    inner(stream, location).unwrap_or(fallback_span)
}
