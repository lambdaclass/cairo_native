use core::{NullableTrait, match_nullable, null, nullable::FromNullableResult};

            fn run_test(x: Option<u8>) -> Option<u8> {
                let a = match x {
                    Option::Some(x) => @NullableTrait::new(x),
                    Option::None(_) => @null::<u8>(),
                };
                let b = *a;
                match match_nullable(b) {
                    FromNullableResult::Null(_) => Option::None(()),
                    FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
                }
            }
