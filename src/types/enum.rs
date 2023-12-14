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
//! | 2 or 3             | `i2`              | `u8`                |
//! | 4, 5, 6 or 7       | `i3`              | `u8`                |
//! | 8 to 15            | `i4`              | `u8`                |
//! | 128 to 255         | `i8`              | `u8`                |
//! | 256 to 511         | `i9`              | `u16`               |
//! | 32768 to 65535     | `i16`             | `u16`               |
//! | 65536 to 131071    | `i17`             | `u32`               |
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
    error::types::{Error, Result},
    metadata::MetadataStorage,
    utils::{get_integer_layout, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{enm::EnumConcreteType, GenericLibfunc, GenericType},
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
pub fn build<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<EnumConcreteType>,
) -> Result<Type<'ctx>>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
    let (_, (tag_ty, tag_layout), variant_tys) =
        get_type_for_variants(context, module, registry, metadata, &info.variants)?;

    let (variant_ty, variant_layout) = variant_tys
        .iter()
        .copied()
        .max_by_key(|(_, layout)| layout.align())
        .unwrap_or((
            llvm::r#type::r#struct(context, &[], false),
            Layout::new::<()>(),
        ));

    let filling_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        (tag_layout.extend(variant_layout)?.1 - tag_layout.size()).try_into()?,
    );

    let total_len = variant_tys
        .iter()
        .map(|(_, layout)| tag_layout.extend(*layout).map(|(x, _)| x.size()))
        .try_fold(0, |acc, x| x.map(|x| acc.max(x)))?;
    let padding_ty = llvm::r#type::array(
        IntegerType::new(context, 8).into(),
        (total_len - tag_layout.extend(variant_layout)?.0.size()).try_into()?,
    );

    Ok(llvm::r#type::r#struct(
        context,
        &[tag_ty, filling_ty, variant_ty, padding_ty],
        false,
    ))
}

/// Extract layout for the default enum representation, its discriminant and all its payloads.
pub fn get_layout_for_variants<TType, TLibfunc>(
    registry: &ProgramRegistry<TType, TLibfunc>,
    variants: &[ConcreteTypeId],
) -> Result<(Layout, Layout, Vec<Layout>)>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
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
pub fn get_type_for_variants<'ctx, TType, TLibfunc>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<TType, TLibfunc>,
    metadata: &mut MetadataStorage,
    variants: &[ConcreteTypeId],
) -> Result<(Layout, TypeLayout<'ctx>, Vec<TypeLayout<'ctx>>)>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = Error>,
{
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
