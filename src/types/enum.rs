//! # Enum type
//!
//! Enumerations are special because they can potentially represent an unlimited amount of things at
//! the same time. They are similar to Rust enums since they can contain data along with the
//! discriminator.
//!
//! ## Layout
//!
//! | Index | Type                 | Description              |
//! | ----- | -------------------- | ------------------------ |
//! |   0   | `iN`                 | Discriminant.            |
//! |   1   | Depends on variants. | Payload.                 |
//!
//! As seen in the table above, an enum's layout is not as simple as concatenating the discriminant
//! with the payload.
//!
//! The discriminant will have the bit width required to store all possible values. The following
//! table contains an example of some number of variants with their discriminant type:
//!
//! | Number of variants | Discriminant type | ABI (in Rust types) |
//! | ------------------ | ----------------- | ------------------- |
//! | 0 or 1             | `i0`              | `()`                |
//! | 2                  | `i1`              | `u8`                |
//! | 3 or 4             | `i2`              | `u8`                |
//! | 5, 6, 7 or 8       | `i3`              | `u8`                |
//! | 9 to 16            | `i4`              | `u8`                |
//! | 129 to 256         | `i8`              | `u8`                |
//! | 257 to 512         | `i9`              | `u16`               |
//! | 32769 to 65536     | `i16`             | `u16`               |
//! | 65537 to 131072    | `i17`             | `u32`               |
//!
//! In Rust, the number of bits and bytes required can be obtained using the following formula:
//!
//! <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
//!  <semantics>
//!   <mtable>
//!    <mtr>
//!     <mtd>
//!      <mrow>
//!       <msub>
//!        <mi>n</mi>
//!        <mi mathvariant="italic">bits</mi>
//!       </msub>
//!       <mo stretchy="false">=</mo>
//!       <mrow>
//!        <mo fence="true" form="prefix" stretchy="true">{</mo>
//!        <mrow>
//!         <mtable>
//!          <mtr>
//!           <mtd>
//!            <mn>0</mn>
//!           </mtd>
//!           <mtd>
//!            <mtext>if</mtext>
//!           </mtd>
//!           <mtd>
//!            <mrow>
//!             <msub>
//!              <mi>n</mi>
//!              <mi mathvariant="italic">variants</mi>
//!             </msub>
//!             <mo stretchy="false">=</mo>
//!             <mn>0</mn>
//!            </mrow>
//!           </mtd>
//!          </mtr>
//!          <mtr>
//!           <mtd>
//!            <mrow>
//!             <mo fence="true" form="prefix" stretchy="true">⌈</mo>
//!             <mrow>
//!              <mrow>
//!               <msub>
//!                <mi>log</mi>
//!                <mn>2</mn>
//!               </msub>
//!               <msub>
//!                <mi>n</mi>
//!                <mi mathvariant="italic">variants</mi>
//!               </msub>
//!              </mrow>
//!             </mrow>
//!             <mo fence="true" form="postfix" stretchy="true">⌉</mo>
//!            </mrow>
//!           </mtd>
//!           <mtd>
//!            <mtext>if</mtext>
//!           </mtd>
//!           <mtd>
//!            <mrow>
//!             <msub>
//!              <mi>n</mi>
//!              <mi mathvariant="italic">variants</mi>
//!             </msub>
//!             <mo stretchy="false">≠</mo>
//!             <mn>0</mn>
//!            </mrow>
//!           </mtd>
//!          </mtr>
//!         </mtable>
//!        </mrow>
//!       </mrow>
//!      </mrow>
//!     </mtd>
//!    </mtr>
//!    <mtr>
//!     <mtd>
//!      <mrow>
//!       <msub>
//!        <mi>n</mi>
//!        <mi mathvariant="italic">bytes</mi>
//!       </msub>
//!       <mo stretchy="false">=</mo>
//!       <mrow>
//!        <mo fence="true" form="prefix" stretchy="true">⌈</mo>
//!        <mrow>
//!         <mfrac>
//!          <msub>
//!           <mi>n</mi>
//!           <mi mathvariant="italic">bits</mi>
//!          </msub>
//!          <mn>8</mn>
//!         </mfrac>
//!        </mrow>
//!        <mo fence="true" form="postfix" stretchy="true">⌉</mo>
//!       </mrow>
//!      </mrow>
//!     </mtd>
//!    </mtr>
//!   </mtable>
//!  </semantics>
//! </math>
//!
//! The payload will then be appended to the discriminant after applying its alignment rules. This
//! will cause unused space between the tag and the payload in most cases. As an example, the
//! following enum will have the layouts described in the table below.
//!
//! ```cairo
//! enum MyEnum {
//!     U8: u8,
//!     U16: u16,
//!     U32: u32,
//!     U64: u64,
//!     Felt: Felt,
//! }
//! ```
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th colspan="6"><code>MyEnum::U8</code></th>
//!         </tr>
//!         <tr>
//!             <th>Index</th>
//!             <th>Type</th>
//!             <th>ABI (in Rust types)</th>
//!             <th>Alignment</th>
//!             <th>Size</th>
//!             <th>Description</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>0</td>
//!             <td><code>i3</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Discriminant.</td>
//!         </tr>
//!         <tr>
//!             <td>1</td>
//!             <td><code>i8</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Payload.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 38]</code></td>
//!             <td>1</td>
//!             <td>38</td>
//!             <td>Padding.</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th colspan="6"><code>MyEnum::U16</code></th>
//!         </tr>
//!         <tr>
//!             <th>Index</th>
//!             <th>Type</th>
//!             <th>ABI (in Rust types)</th>
//!             <th>Alignment</th>
//!             <th>Size</th>
//!             <th>Description</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>0</td>
//!             <td><code>i3</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Discriminant.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 1]</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Padding.</td>
//!         </tr>
//!         <tr>
//!             <td>1</td>
//!             <td><code>i16</code></td>
//!             <td><code>u16</code></td>
//!             <td>2</td>
//!             <td>2</td>
//!             <td>Payload.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 36]</code></td>
//!             <td>1</td>
//!             <td>36</td>
//!             <td>Padding.</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th colspan="6"><code>MyEnum::U32</code></th>
//!         </tr>
//!         <tr>
//!             <th>Index</th>
//!             <th>Type</th>
//!             <th>ABI (in Rust types)</th>
//!             <th>Alignment</th>
//!             <th>Size</th>
//!             <th>Description</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>0</td>
//!             <td><code>i3</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Discriminant.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 3]</code></td>
//!             <td>1</td>
//!             <td>3</td>
//!             <td>Padding.</td>
//!         </tr>
//!         <tr>
//!             <td>1</td>
//!             <td><code>i32</code></td>
//!             <td><code>u32</code></td>
//!             <td>4</td>
//!             <td>4</td>
//!             <td>Payload.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 32]</code></td>
//!             <td>1</td>
//!             <td>32</td>
//!             <td>Padding.</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th colspan="6"><code>MyEnum::U64</code></th>
//!         </tr>
//!         <tr>
//!             <th>Index</th>
//!             <th>Type</th>
//!             <th>ABI (in Rust types)</th>
//!             <th>Alignment</th>
//!             <th>Size</th>
//!             <th>Description</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>0</td>
//!             <td><code>i3</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Discriminant.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 7]</code></td>
//!             <td>1</td>
//!             <td>7</td>
//!             <td>Padding.</td>
//!         </tr>
//!         <tr>
//!             <td>1</td>
//!             <td><code>i64</code></td>
//!             <td><code>u64</code></td>
//!             <td>8</td>
//!             <td>8</td>
//!             <td>Payload.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 24]</code></td>
//!             <td>1</td>
//!             <td>24</td>
//!             <td>Padding.</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! <table>
//!     <thead>
//!         <tr>
//!             <th colspan="6"><code>MyEnum</code></th>
//!         </tr>
//!         <tr>
//!             <th>Index</th>
//!             <th>Type</th>
//!             <th>ABI (in Rust types)</th>
//!             <th>Alignment</th>
//!             <th>Size</th>
//!             <th>Description</th>
//!         </tr>
//!     </thead>
//!     <tbody>
//!         <tr>
//!             <td>0</td>
//!             <td><code>i3</code></td>
//!             <td><code>u8</code></td>
//!             <td>1</td>
//!             <td>1</td>
//!             <td>Discriminant.</td>
//!         </tr>
//!         <tr>
//!             <td>N/A</td>
//!             <td>N/A</td>
//!             <td><code>[u8; 7]</code></td>
//!             <td>1</td>
//!             <td>7</td>
//!             <td>Padding.</td>
//!         </tr>
//!         <tr>
//!             <td>1</td>
//!             <td><code>i252</code></td>
//!             <td><code>[u64; 4]</code></td>
//!             <td>8</td>
//!             <td>32</td>
//!             <td>Payload.</td>
//!         </tr>
//!     </tbody>
//! </table>
//!
//! As seen above, while the discriminant is always at the same offset, the payloads don't necessary
//! have the same offset between all variants. It depends on the payload's alignment.
//!
//! In reality, the first variant will have a zero-sized padding between the discriminant and the
//! payload to keep everything consistent and the padding will have its own index, shifting every
//! index below it by one. However all that's been ignored for documenting purposes.
//!
//! An MLIR type cannot be an enumeration (it doesn't exist), therefore a variant or a buffer has to
//! be used. Using a buffer as a dummy payload has been discarded because it doesn't keep the enum's
//! alignment information. To keep that info, the first variant with the biggest alignment is used
//! as the default payload.
//!
//! Using the info stated above, we can infer that the example enum will have the following type by
//! default:
//!
//! | Index | Type  | ABI (in Rust types) | Alignment | Size | Description   |
//! | ----- | ----- | ------------------- | --------- | ---- | ------------- |
//! |   0   | `i3`  | `u8`                |         1 |    1 | Discriminant. |
//! |  N/A  | N/A   | `[u8; 7]`           |         1 |    7 | Padding.      |
//! |   1   | `i64` | `u64`               |         8 |    8 | Payload.      |
//! |  N/A  | N/A   | `[u8; 24]`          |         1 |   24 | Padding.      |

use super::{TypeBuilder, WithSelf};
use crate::{
    error::builders::Result,
    metadata::MetadataStorage,
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        enm::EnumConcreteType,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};
use std::alloc::Layout;

/// An MLIR type with its memory layout.
pub type TypeLayout<'ctx> = (Type<'ctx>, Layout);

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<EnumConcreteType>,
) -> Result<Type<'ctx>> {
    let tag_bits = info.variants.len().next_power_of_two().trailing_zeros();

    let tag_layout = get_integer_layout(tag_bits);
    let layout = info.variants.iter().fold(tag_layout, |acc, id| {
        let layout = tag_layout
            .extend(registry.get_type(id).unwrap().layout(registry).unwrap())
            .unwrap()
            .0;

        Layout::from_size_align(
            acc.size().max(layout.size()),
            acc.align().max(layout.align()),
        )
        .unwrap()
    });

    let i8_ty = IntegerType::new(context, 8).into();
    Ok(if layout.size() == 0 {
        llvm::r#type::array(i8_ty, 0)
    } else {
        match info.variants.len() {
            0 => unreachable!(),
            1 => registry.build_type(context, module, registry, metadata, &info.variants[0])?,
            _ if info
                .variants
                .iter()
                .all(|type_id| registry.get_type(type_id).unwrap().is_zst(registry)) =>
            {
                llvm::r#type::r#struct(
                    context,
                    &[
                        IntegerType::new(context, tag_bits).into(),
                        llvm::r#type::array(i8_ty, 0),
                    ],
                    false,
                )
            }
            _ => llvm::r#type::r#struct(
                context,
                &[
                    IntegerType::new(context, (8 * layout.align()) as u32).into(),
                    llvm::r#type::array(i8_ty, (layout.size() - layout.align()) as u32),
                ],
                false,
            ),
        }
    })
}

/// Extract layout for the default enum representation, its discriminant and all its payloads.
pub fn get_layout_for_variants(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    variants: &[ConcreteTypeId],
) -> Result<(Layout, Layout, Vec<Layout>)> {
    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
    let tag_layout = get_integer_layout(tag_bits);

    let mut layout = tag_layout;
    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let concrete_payload_ty = registry.get_type(variant)?;
        let payload_layout = concrete_payload_ty.layout(registry)?;

        let full_layout = tag_layout.extend(payload_layout)?.0;
        layout = Layout::from_size_align(
            layout.size().max(full_layout.size()),
            layout.align().max(full_layout.align()),
        )?;

        output.push(payload_layout);
    }

    Ok((layout, tag_layout, output))
}

/// Extract the type and layout for the default enum representation, its discriminant and all its
/// payloads.
// TODO: Change this function to accept a slice of slices (for variants). Not all uses have a slice
//   with one `ConcreteTypeId` per variant (deploy_syscalls has two types for the Ok() variant).
pub fn get_type_for_variants<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    variants: &[ConcreteTypeId],
) -> Result<(Layout, TypeLayout<'ctx>, Vec<TypeLayout<'ctx>>)> {
    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
    let tag_layout = get_integer_layout(tag_bits);
    let tag_ty: Type = IntegerType::new(context, tag_bits).into();

    let mut layout = tag_layout;
    let mut output = Vec::with_capacity(variants.len());
    for variant in variants {
        let (payload_ty, payload_layout) =
            registry.build_type_with_layout(context, module, registry, metadata, variant)?;

        let full_layout = tag_layout.extend(payload_layout)?.0;
        layout = Layout::from_size_align(
            layout.size().max(full_layout.size()),
            layout.align().max(full_layout.align()),
        )?;

        output.push((payload_ty, payload_layout));
    }

    Ok((layout, (tag_ty, tag_layout), output))
}

#[cfg(test)]
mod test {
    use crate::{metadata::MetadataStorage, types::TypeBuilder, utils::test::load_cairo};
    use cairo_lang_sierra::{
        extensions::core::{CoreLibfunc, CoreType},
        program_registry::ProgramRegistry,
    };
    use melior::{
        ir::{r#type::IntegerType, Location, Module},
        Context,
    };

    #[test]
    fn enum_type_single_variant_no_i0() {
        let (_, program) = load_cairo! {
            enum MyEnum {
                A: felt252,
            }

            fn run_program(x: MyEnum) -> MyEnum {
                x
            }
        };

        let context = Context::new();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        let module = Module::new(Location::unknown(&context));
        let mut metadata = MetadataStorage::new();

        let i0_ty = IntegerType::new(&context, 0).into();
        program
            .type_declarations
            .iter()
            .map(|ty| (&ty.id, registry.get_type(&ty.id).unwrap()))
            .map(|(id, ty)| {
                ty.build(&context, &module, &registry, &mut metadata, id)
                    .unwrap()
            })
            .any(|width| width == i0_ty);
    }
}
