////! # Enum type
//! # Enum type
////!
//!
////! Enumerations are special because they can potentially represent an unlimited amount of things at
//! Enumerations are special because they can potentially represent an unlimited amount of things at
////! the same time. They are similar to Rust enums since they can contain data along with the
//! the same time. They are similar to Rust enums since they can contain data along with the
////! discriminator.
//! discriminator.
////!
//!
////! ## Layout
//! ## Layout
////!
//!
////! | Index | Type                 | Description              |
//! | Index | Type                 | Description              |
////! | ----- | -------------------- | ------------------------ |
//! | ----- | -------------------- | ------------------------ |
////! |   0   | `iN`                 | Discriminant.            |
//! |   0   | `iN`                 | Discriminant.            |
////! |   1   | Depends on variants. | Payload.                 |
//! |   1   | Depends on variants. | Payload.                 |
////!
//!
////! As seen in the table above, an enum's layout is not as simple as concatenating the discriminant
//! As seen in the table above, an enum's layout is not as simple as concatenating the discriminant
////! with the payload.
//! with the payload.
////!
//!
////! The discriminant will have the bit width required to store all possible values. The following
//! The discriminant will have the bit width required to store all possible values. The following
////! table contains an example of some number of variants with their discriminant type:
//! table contains an example of some number of variants with their discriminant type:
////!
//!
////! | Number of variants | Discriminant type | ABI (in Rust types) |
//! | Number of variants | Discriminant type | ABI (in Rust types) |
////! | ------------------ | ----------------- | ------------------- |
//! | ------------------ | ----------------- | ------------------- |
////! | 0 or 1             | `i0`              | `()`                |
//! | 0 or 1             | `i0`              | `()`                |
////! | 2                  | `i1`              | `u8`                |
//! | 2                  | `i1`              | `u8`                |
////! | 3 or 4             | `i2`              | `u8`                |
//! | 3 or 4             | `i2`              | `u8`                |
////! | 5, 6, 7 or 8       | `i3`              | `u8`                |
//! | 5, 6, 7 or 8       | `i3`              | `u8`                |
////! | 9 to 16            | `i4`              | `u8`                |
//! | 9 to 16            | `i4`              | `u8`                |
////! | 129 to 256         | `i8`              | `u8`                |
//! | 129 to 256         | `i8`              | `u8`                |
////! | 257 to 512         | `i9`              | `u16`               |
//! | 257 to 512         | `i9`              | `u16`               |
////! | 32769 to 65536     | `i16`             | `u16`               |
//! | 32769 to 65536     | `i16`             | `u16`               |
////! | 65537 to 131072    | `i17`             | `u32`               |
//! | 65537 to 131072    | `i17`             | `u32`               |
////!
//!
////! In Rust, the number of bits and bytes required can be obtained using the following formula:
//! In Rust, the number of bits and bytes required can be obtained using the following formula:
////!
//!
////! <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
//! <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
////!  <semantics>
//!  <semantics>
////!   <mtable>
//!   <mtable>
////!    <mtr>
//!    <mtr>
////!     <mtd>
//!     <mtd>
////!      <mrow>
//!      <mrow>
////!       <msub>
//!       <msub>
////!        <mi>n</mi>
//!        <mi>n</mi>
////!        <mi mathvariant="italic">bits</mi>
//!        <mi mathvariant="italic">bits</mi>
////!       </msub>
//!       </msub>
////!       <mo stretchy="false">=</mo>
//!       <mo stretchy="false">=</mo>
////!       <mrow>
//!       <mrow>
////!        <mo fence="true" form="prefix" stretchy="true">{</mo>
//!        <mo fence="true" form="prefix" stretchy="true">{</mo>
////!        <mrow>
//!        <mrow>
////!         <mtable>
//!         <mtable>
////!          <mtr>
//!          <mtr>
////!           <mtd>
//!           <mtd>
////!            <mn>0</mn>
//!            <mn>0</mn>
////!           </mtd>
//!           </mtd>
////!           <mtd>
//!           <mtd>
////!            <mtext>if</mtext>
//!            <mtext>if</mtext>
////!           </mtd>
//!           </mtd>
////!           <mtd>
//!           <mtd>
////!            <mrow>
//!            <mrow>
////!             <msub>
//!             <msub>
////!              <mi>n</mi>
//!              <mi>n</mi>
////!              <mi mathvariant="italic">variants</mi>
//!              <mi mathvariant="italic">variants</mi>
////!             </msub>
//!             </msub>
////!             <mo stretchy="false">=</mo>
//!             <mo stretchy="false">=</mo>
////!             <mn>0</mn>
//!             <mn>0</mn>
////!            </mrow>
//!            </mrow>
////!           </mtd>
//!           </mtd>
////!          </mtr>
//!          </mtr>
////!          <mtr>
//!          <mtr>
////!           <mtd>
//!           <mtd>
////!            <mrow>
//!            <mrow>
////!             <mo fence="true" form="prefix" stretchy="true">⌈</mo>
//!             <mo fence="true" form="prefix" stretchy="true">⌈</mo>
////!             <mrow>
//!             <mrow>
////!              <mrow>
//!              <mrow>
////!               <msub>
//!               <msub>
////!                <mi>log</mi>
//!                <mi>log</mi>
////!                <mn>2</mn>
//!                <mn>2</mn>
////!               </msub>
//!               </msub>
////!               <msub>
//!               <msub>
////!                <mi>n</mi>
//!                <mi>n</mi>
////!                <mi mathvariant="italic">variants</mi>
//!                <mi mathvariant="italic">variants</mi>
////!               </msub>
//!               </msub>
////!              </mrow>
//!              </mrow>
////!             </mrow>
//!             </mrow>
////!             <mo fence="true" form="postfix" stretchy="true">⌉</mo>
//!             <mo fence="true" form="postfix" stretchy="true">⌉</mo>
////!            </mrow>
//!            </mrow>
////!           </mtd>
//!           </mtd>
////!           <mtd>
//!           <mtd>
////!            <mtext>if</mtext>
//!            <mtext>if</mtext>
////!           </mtd>
//!           </mtd>
////!           <mtd>
//!           <mtd>
////!            <mrow>
//!            <mrow>
////!             <msub>
//!             <msub>
////!              <mi>n</mi>
//!              <mi>n</mi>
////!              <mi mathvariant="italic">variants</mi>
//!              <mi mathvariant="italic">variants</mi>
////!             </msub>
//!             </msub>
////!             <mo stretchy="false">≠</mo>
//!             <mo stretchy="false">≠</mo>
////!             <mn>0</mn>
//!             <mn>0</mn>
////!            </mrow>
//!            </mrow>
////!           </mtd>
//!           </mtd>
////!          </mtr>
//!          </mtr>
////!         </mtable>
//!         </mtable>
////!        </mrow>
//!        </mrow>
////!       </mrow>
//!       </mrow>
////!      </mrow>
//!      </mrow>
////!     </mtd>
//!     </mtd>
////!    </mtr>
//!    </mtr>
////!    <mtr>
//!    <mtr>
////!     <mtd>
//!     <mtd>
////!      <mrow>
//!      <mrow>
////!       <msub>
//!       <msub>
////!        <mi>n</mi>
//!        <mi>n</mi>
////!        <mi mathvariant="italic">bytes</mi>
//!        <mi mathvariant="italic">bytes</mi>
////!       </msub>
//!       </msub>
////!       <mo stretchy="false">=</mo>
//!       <mo stretchy="false">=</mo>
////!       <mrow>
//!       <mrow>
////!        <mo fence="true" form="prefix" stretchy="true">⌈</mo>
//!        <mo fence="true" form="prefix" stretchy="true">⌈</mo>
////!        <mrow>
//!        <mrow>
////!         <mfrac>
//!         <mfrac>
////!          <msub>
//!          <msub>
////!           <mi>n</mi>
//!           <mi>n</mi>
////!           <mi mathvariant="italic">bits</mi>
//!           <mi mathvariant="italic">bits</mi>
////!          </msub>
//!          </msub>
////!          <mn>8</mn>
//!          <mn>8</mn>
////!         </mfrac>
//!         </mfrac>
////!        </mrow>
//!        </mrow>
////!        <mo fence="true" form="postfix" stretchy="true">⌉</mo>
//!        <mo fence="true" form="postfix" stretchy="true">⌉</mo>
////!       </mrow>
//!       </mrow>
////!      </mrow>
//!      </mrow>
////!     </mtd>
//!     </mtd>
////!    </mtr>
//!    </mtr>
////!   </mtable>
//!   </mtable>
////!  </semantics>
//!  </semantics>
////! </math>
//! </math>
////!
//!
////! The payload will then be appended to the discriminant after applying its alignment rules. This
//! The payload will then be appended to the discriminant after applying its alignment rules. This
////! will cause unused space between the tag and the payload in most cases. As an example, the
//! will cause unused space between the tag and the payload in most cases. As an example, the
////! following enum will have the layouts described in the table below.
//! following enum will have the layouts described in the table below.
////!
//!
////! ```cairo
//! ```cairo
////! enum MyEnum {
//! enum MyEnum {
////!     U8: u8,
//!     U8: u8,
////!     U16: u16,
//!     U16: u16,
////!     U32: u32,
//!     U32: u32,
////!     U64: u64,
//!     U64: u64,
////!     Felt: Felt,
//!     Felt: Felt,
////! }
//! }
////! ```
//! ```
////!
//!
////! <table>
//! <table>
////!     <thead>
//!     <thead>
////!         <tr>
//!         <tr>
////!             <th colspan="6"><code>MyEnum::U8</code></th>
//!             <th colspan="6"><code>MyEnum::U8</code></th>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <th>Index</th>
//!             <th>Index</th>
////!             <th>Type</th>
//!             <th>Type</th>
////!             <th>ABI (in Rust types)</th>
//!             <th>ABI (in Rust types)</th>
////!             <th>Alignment</th>
//!             <th>Alignment</th>
////!             <th>Size</th>
//!             <th>Size</th>
////!             <th>Description</th>
//!             <th>Description</th>
////!         </tr>
//!         </tr>
////!     </thead>
//!     </thead>
////!     <tbody>
//!     <tbody>
////!         <tr>
//!         <tr>
////!             <td>0</td>
//!             <td>0</td>
////!             <td><code>i3</code></td>
//!             <td><code>i3</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Discriminant.</td>
//!             <td>Discriminant.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>1</td>
//!             <td>1</td>
////!             <td><code>i8</code></td>
//!             <td><code>i8</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Payload.</td>
//!             <td>Payload.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 38]</code></td>
//!             <td><code>[u8; 38]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>38</td>
//!             <td>38</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!     </tbody>
//!     </tbody>
////! </table>
//! </table>
////!
//!
////! <table>
//! <table>
////!     <thead>
//!     <thead>
////!         <tr>
//!         <tr>
////!             <th colspan="6"><code>MyEnum::U16</code></th>
//!             <th colspan="6"><code>MyEnum::U16</code></th>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <th>Index</th>
//!             <th>Index</th>
////!             <th>Type</th>
//!             <th>Type</th>
////!             <th>ABI (in Rust types)</th>
//!             <th>ABI (in Rust types)</th>
////!             <th>Alignment</th>
//!             <th>Alignment</th>
////!             <th>Size</th>
//!             <th>Size</th>
////!             <th>Description</th>
//!             <th>Description</th>
////!         </tr>
//!         </tr>
////!     </thead>
//!     </thead>
////!     <tbody>
//!     <tbody>
////!         <tr>
//!         <tr>
////!             <td>0</td>
//!             <td>0</td>
////!             <td><code>i3</code></td>
//!             <td><code>i3</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Discriminant.</td>
//!             <td>Discriminant.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 1]</code></td>
//!             <td><code>[u8; 1]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>1</td>
//!             <td>1</td>
////!             <td><code>i16</code></td>
//!             <td><code>i16</code></td>
////!             <td><code>u16</code></td>
//!             <td><code>u16</code></td>
////!             <td>2</td>
//!             <td>2</td>
////!             <td>2</td>
//!             <td>2</td>
////!             <td>Payload.</td>
//!             <td>Payload.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 36]</code></td>
//!             <td><code>[u8; 36]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>36</td>
//!             <td>36</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!     </tbody>
//!     </tbody>
////! </table>
//! </table>
////!
//!
////! <table>
//! <table>
////!     <thead>
//!     <thead>
////!         <tr>
//!         <tr>
////!             <th colspan="6"><code>MyEnum::U32</code></th>
//!             <th colspan="6"><code>MyEnum::U32</code></th>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <th>Index</th>
//!             <th>Index</th>
////!             <th>Type</th>
//!             <th>Type</th>
////!             <th>ABI (in Rust types)</th>
//!             <th>ABI (in Rust types)</th>
////!             <th>Alignment</th>
//!             <th>Alignment</th>
////!             <th>Size</th>
//!             <th>Size</th>
////!             <th>Description</th>
//!             <th>Description</th>
////!         </tr>
//!         </tr>
////!     </thead>
//!     </thead>
////!     <tbody>
//!     <tbody>
////!         <tr>
//!         <tr>
////!             <td>0</td>
//!             <td>0</td>
////!             <td><code>i3</code></td>
//!             <td><code>i3</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Discriminant.</td>
//!             <td>Discriminant.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 3]</code></td>
//!             <td><code>[u8; 3]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>3</td>
//!             <td>3</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>1</td>
//!             <td>1</td>
////!             <td><code>i32</code></td>
//!             <td><code>i32</code></td>
////!             <td><code>u32</code></td>
//!             <td><code>u32</code></td>
////!             <td>4</td>
//!             <td>4</td>
////!             <td>4</td>
//!             <td>4</td>
////!             <td>Payload.</td>
//!             <td>Payload.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 32]</code></td>
//!             <td><code>[u8; 32]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>32</td>
//!             <td>32</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!     </tbody>
//!     </tbody>
////! </table>
//! </table>
////!
//!
////! <table>
//! <table>
////!     <thead>
//!     <thead>
////!         <tr>
//!         <tr>
////!             <th colspan="6"><code>MyEnum::U64</code></th>
//!             <th colspan="6"><code>MyEnum::U64</code></th>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <th>Index</th>
//!             <th>Index</th>
////!             <th>Type</th>
//!             <th>Type</th>
////!             <th>ABI (in Rust types)</th>
//!             <th>ABI (in Rust types)</th>
////!             <th>Alignment</th>
//!             <th>Alignment</th>
////!             <th>Size</th>
//!             <th>Size</th>
////!             <th>Description</th>
//!             <th>Description</th>
////!         </tr>
//!         </tr>
////!     </thead>
//!     </thead>
////!     <tbody>
//!     <tbody>
////!         <tr>
//!         <tr>
////!             <td>0</td>
//!             <td>0</td>
////!             <td><code>i3</code></td>
//!             <td><code>i3</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Discriminant.</td>
//!             <td>Discriminant.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 7]</code></td>
//!             <td><code>[u8; 7]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>7</td>
//!             <td>7</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>1</td>
//!             <td>1</td>
////!             <td><code>i64</code></td>
//!             <td><code>i64</code></td>
////!             <td><code>u64</code></td>
//!             <td><code>u64</code></td>
////!             <td>8</td>
//!             <td>8</td>
////!             <td>8</td>
//!             <td>8</td>
////!             <td>Payload.</td>
//!             <td>Payload.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 24]</code></td>
//!             <td><code>[u8; 24]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>24</td>
//!             <td>24</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!     </tbody>
//!     </tbody>
////! </table>
//! </table>
////!
//!
////! <table>
//! <table>
////!     <thead>
//!     <thead>
////!         <tr>
//!         <tr>
////!             <th colspan="6"><code>MyEnum</code></th>
//!             <th colspan="6"><code>MyEnum</code></th>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <th>Index</th>
//!             <th>Index</th>
////!             <th>Type</th>
//!             <th>Type</th>
////!             <th>ABI (in Rust types)</th>
//!             <th>ABI (in Rust types)</th>
////!             <th>Alignment</th>
//!             <th>Alignment</th>
////!             <th>Size</th>
//!             <th>Size</th>
////!             <th>Description</th>
//!             <th>Description</th>
////!         </tr>
//!         </tr>
////!     </thead>
//!     </thead>
////!     <tbody>
//!     <tbody>
////!         <tr>
//!         <tr>
////!             <td>0</td>
//!             <td>0</td>
////!             <td><code>i3</code></td>
//!             <td><code>i3</code></td>
////!             <td><code>u8</code></td>
//!             <td><code>u8</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>Discriminant.</td>
//!             <td>Discriminant.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td>N/A</td>
//!             <td>N/A</td>
////!             <td><code>[u8; 7]</code></td>
//!             <td><code>[u8; 7]</code></td>
////!             <td>1</td>
//!             <td>1</td>
////!             <td>7</td>
//!             <td>7</td>
////!             <td>Padding.</td>
//!             <td>Padding.</td>
////!         </tr>
//!         </tr>
////!         <tr>
//!         <tr>
////!             <td>1</td>
//!             <td>1</td>
////!             <td><code>i252</code></td>
//!             <td><code>i252</code></td>
////!             <td><code>[u64; 4]</code></td>
//!             <td><code>[u64; 4]</code></td>
////!             <td>8</td>
//!             <td>8</td>
////!             <td>32</td>
//!             <td>32</td>
////!             <td>Payload.</td>
//!             <td>Payload.</td>
////!         </tr>
//!         </tr>
////!     </tbody>
//!     </tbody>
////! </table>
//! </table>
////!
//!
////! As seen above, while the discriminant is always at the same offset, the payloads don't necessary
//! As seen above, while the discriminant is always at the same offset, the payloads don't necessary
////! have the same offset between all variants. It depends on the payload's alignment.
//! have the same offset between all variants. It depends on the payload's alignment.
////!
//!
////! In reality, the first variant will have a zero-sized padding between the discriminant and the
//! In reality, the first variant will have a zero-sized padding between the discriminant and the
////! payload to keep everything consistent and the padding will have its own index, shifting every
//! payload to keep everything consistent and the padding will have its own index, shifting every
////! index below it by one. However all that's been ignored for documenting purposes.
//! index below it by one. However all that's been ignored for documenting purposes.
////!
//!
////! An MLIR type cannot be an enumeration (it doesn't exist), therefore a variant or a buffer has to
//! An MLIR type cannot be an enumeration (it doesn't exist), therefore a variant or a buffer has to
////! be used. Using a buffer as a dummy payload has been discarded because it doesn't keep the enum's
//! be used. Using a buffer as a dummy payload has been discarded because it doesn't keep the enum's
////! alignment information. To keep that info, the first variant with the biggest alignment is used
//! alignment information. To keep that info, the first variant with the biggest alignment is used
////! as the default payload.
//! as the default payload.
////!
//!
////! Using the info stated above, we can infer that the example enum will have the following type by
//! Using the info stated above, we can infer that the example enum will have the following type by
////! default:
//! default:
////!
//!
////! | Index | Type  | ABI (in Rust types) | Alignment | Size | Description   |
//! | Index | Type  | ABI (in Rust types) | Alignment | Size | Description   |
////! | ----- | ----- | ------------------- | --------- | ---- | ------------- |
//! | ----- | ----- | ------------------- | --------- | ---- | ------------- |
////! |   0   | `i3`  | `u8`                |         1 |    1 | Discriminant. |
//! |   0   | `i3`  | `u8`                |         1 |    1 | Discriminant. |
////! |  N/A  | N/A   | `[u8; 7]`           |         1 |    7 | Padding.      |
//! |  N/A  | N/A   | `[u8; 7]`           |         1 |    7 | Padding.      |
////! |   1   | `i64` | `u64`               |         8 |    8 | Payload.      |
//! |   1   | `i64` | `u64`               |         8 |    8 | Payload.      |
////! |  N/A  | N/A   | `[u8; 24]`          |         1 |   24 | Padding.      |
//! |  N/A  | N/A   | `[u8; 24]`          |         1 |   24 | Padding.      |
//

//use super::{TypeBuilder, WithSelf};
use super::{TypeBuilder, WithSelf};
//use crate::{
use crate::{
//    error::Result,
    error::Result,
//    metadata::MetadataStorage,
    metadata::MetadataStorage,
//    utils::{get_integer_layout, ProgramRegistryExt},
    utils::{get_integer_layout, ProgramRegistryExt},
//};
};
//use cairo_lang_sierra::{
use cairo_lang_sierra::{
//    extensions::{
    extensions::{
//        core::{CoreLibfunc, CoreType},
        core::{CoreLibfunc, CoreType},
//        enm::EnumConcreteType,
        enm::EnumConcreteType,
//    },
    },
//    ids::ConcreteTypeId,
    ids::ConcreteTypeId,
//    program_registry::ProgramRegistry,
    program_registry::ProgramRegistry,
//};
};
//use melior::{
use melior::{
//    dialect::llvm,
    dialect::llvm,
//    ir::{r#type::IntegerType, Module, Type},
    ir::{r#type::IntegerType, Module, Type},
//    Context,
    Context,
//};
};
//use std::alloc::Layout;
use std::alloc::Layout;
//

///// An MLIR type with its memory layout.
/// An MLIR type with its memory layout.
//pub type TypeLayout<'ctx> = (Type<'ctx>, Layout);
pub type TypeLayout<'ctx> = (Type<'ctx>, Layout);
//

///// Build the MLIR type.
/// Build the MLIR type.
/////
///
///// Check out [the module](self) for more info.
/// Check out [the module](self) for more info.
//pub fn build<'ctx>(
pub fn build<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    info: WithSelf<EnumConcreteType>,
    info: WithSelf<EnumConcreteType>,
//) -> Result<Type<'ctx>> {
) -> Result<Type<'ctx>> {
//    let tag_bits = info.variants.len().next_power_of_two().trailing_zeros();
    let tag_bits = info.variants.len().next_power_of_two().trailing_zeros();
//

//    let tag_layout = get_integer_layout(tag_bits);
    let tag_layout = get_integer_layout(tag_bits);
//    let layout = info.variants.iter().fold(tag_layout, |acc, id| {
    let layout = info.variants.iter().fold(tag_layout, |acc, id| {
//        let layout = tag_layout
        let layout = tag_layout
//            .extend(registry.get_type(id).unwrap().layout(registry).unwrap())
            .extend(registry.get_type(id).unwrap().layout(registry).unwrap())
//            .unwrap()
            .unwrap()
//            .0;
            .0;
//

//        Layout::from_size_align(
        Layout::from_size_align(
//            acc.size().max(layout.size()),
            acc.size().max(layout.size()),
//            acc.align().max(layout.align()),
            acc.align().max(layout.align()),
//        )
        )
//        .unwrap()
        .unwrap()
//    });
    });
//

//    let i8_ty = IntegerType::new(context, 8).into();
    let i8_ty = IntegerType::new(context, 8).into();
//    Ok(match info.variants.len() {
    Ok(match info.variants.len() {
//        0 => llvm::r#type::array(IntegerType::new(context, 8).into(), 0),
        0 => llvm::r#type::array(IntegerType::new(context, 8).into(), 0),
//        1 => registry.build_type(context, module, registry, metadata, &info.variants[0])?,
        1 => registry.build_type(context, module, registry, metadata, &info.variants[0])?,
//        _ if info
        _ if info
//            .variants
            .variants
//            .iter()
            .iter()
//            .all(|type_id| registry.get_type(type_id).unwrap().is_zst(registry)) =>
            .all(|type_id| registry.get_type(type_id).unwrap().is_zst(registry)) =>
//        {
        {
//            llvm::r#type::r#struct(
            llvm::r#type::r#struct(
//                context,
                context,
//                &[
                &[
//                    IntegerType::new(context, tag_bits).into(),
                    IntegerType::new(context, tag_bits).into(),
//                    llvm::r#type::array(i8_ty, 0),
                    llvm::r#type::array(i8_ty, 0),
//                ],
                ],
//                false,
                false,
//            )
            )
//        }
        }
//        _ => llvm::r#type::r#struct(
        _ => llvm::r#type::r#struct(
//            context,
            context,
//            &[
            &[
//                IntegerType::new(context, (8 * layout.align()) as u32).into(),
                IntegerType::new(context, (8 * layout.align()) as u32).into(),
//                llvm::r#type::array(i8_ty, (layout.size() - layout.align()) as u32),
                llvm::r#type::array(i8_ty, (layout.size() - layout.align()) as u32),
//            ],
            ],
//            false,
            false,
//        ),
        ),
//    })
    })
//}
}
//

///// Extract layout for the default enum representation, its discriminant and all its payloads.
/// Extract layout for the default enum representation, its discriminant and all its payloads.
//pub fn get_layout_for_variants(
pub fn get_layout_for_variants(
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    variants: &[ConcreteTypeId],
    variants: &[ConcreteTypeId],
//) -> Result<(Layout, Layout, Vec<Layout>)> {
) -> Result<(Layout, Layout, Vec<Layout>)> {
//    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
//    let tag_layout = get_integer_layout(tag_bits);
    let tag_layout = get_integer_layout(tag_bits);
//

//    let mut layout = tag_layout;
    let mut layout = tag_layout;
//    let mut output = Vec::with_capacity(variants.len());
    let mut output = Vec::with_capacity(variants.len());
//    for variant in variants {
    for variant in variants {
//        let concrete_payload_ty = registry.get_type(variant)?;
        let concrete_payload_ty = registry.get_type(variant)?;
//        let payload_layout = concrete_payload_ty.layout(registry)?;
        let payload_layout = concrete_payload_ty.layout(registry)?;
//

//        let full_layout = tag_layout.extend(payload_layout)?.0;
        let full_layout = tag_layout.extend(payload_layout)?.0;
//        layout = Layout::from_size_align(
        layout = Layout::from_size_align(
//            layout.size().max(full_layout.size()),
            layout.size().max(full_layout.size()),
//            layout.align().max(full_layout.align()),
            layout.align().max(full_layout.align()),
//        )?;
        )?;
//

//        output.push(payload_layout);
        output.push(payload_layout);
//    }
    }
//

//    Ok((layout, tag_layout, output))
    Ok((layout, tag_layout, output))
//}
}
//

///// Extract the type and layout for the default enum representation, its discriminant and all its
/// Extract the type and layout for the default enum representation, its discriminant and all its
///// payloads.
/// payloads.
//// TODO: Change this function to accept a slice of slices (for variants). Not all uses have a slice
// TODO: Change this function to accept a slice of slices (for variants). Not all uses have a slice
////   with one `ConcreteTypeId` per variant (deploy_syscalls has two types for the Ok() variant).
//   with one `ConcreteTypeId` per variant (deploy_syscalls has two types for the Ok() variant).
//pub fn get_type_for_variants<'ctx>(
pub fn get_type_for_variants<'ctx>(
//    context: &'ctx Context,
    context: &'ctx Context,
//    module: &Module<'ctx>,
    module: &Module<'ctx>,
//    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
//    metadata: &mut MetadataStorage,
    metadata: &mut MetadataStorage,
//    variants: &[ConcreteTypeId],
    variants: &[ConcreteTypeId],
//) -> Result<(Layout, TypeLayout<'ctx>, Vec<TypeLayout<'ctx>>)> {
) -> Result<(Layout, TypeLayout<'ctx>, Vec<TypeLayout<'ctx>>)> {
//    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
    let tag_bits = variants.len().next_power_of_two().trailing_zeros();
//    let tag_layout = get_integer_layout(tag_bits);
    let tag_layout = get_integer_layout(tag_bits);
//    let tag_ty: Type = IntegerType::new(context, tag_bits).into();
    let tag_ty: Type = IntegerType::new(context, tag_bits).into();
//

//    let mut layout = tag_layout;
    let mut layout = tag_layout;
//    let mut output = Vec::with_capacity(variants.len());
    let mut output = Vec::with_capacity(variants.len());
//    for variant in variants {
    for variant in variants {
//        let (payload_ty, payload_layout) =
        let (payload_ty, payload_layout) =
//            registry.build_type_with_layout(context, module, registry, metadata, variant)?;
            registry.build_type_with_layout(context, module, registry, metadata, variant)?;
//

//        let full_layout = tag_layout.extend(payload_layout)?.0;
        let full_layout = tag_layout.extend(payload_layout)?.0;
//        layout = Layout::from_size_align(
        layout = Layout::from_size_align(
//            layout.size().max(full_layout.size()),
            layout.size().max(full_layout.size()),
//            layout.align().max(full_layout.align()),
            layout.align().max(full_layout.align()),
//        )?;
        )?;
//

//        output.push((payload_ty, payload_layout));
        output.push((payload_ty, payload_layout));
//    }
    }
//

//    Ok((layout, (tag_ty, tag_layout), output))
    Ok((layout, (tag_ty, tag_layout), output))
//}
}
//

//#[cfg(test)]
#[cfg(test)]
//mod test {
mod test {
//    use crate::{metadata::MetadataStorage, types::TypeBuilder, utils::test::load_cairo};
    use crate::{metadata::MetadataStorage, types::TypeBuilder, utils::test::load_cairo};
//    use cairo_lang_sierra::{
    use cairo_lang_sierra::{
//        extensions::core::{CoreLibfunc, CoreType},
        extensions::core::{CoreLibfunc, CoreType},
//        program_registry::ProgramRegistry,
        program_registry::ProgramRegistry,
//    };
    };
//    use melior::{
    use melior::{
//        ir::{r#type::IntegerType, Location, Module},
        ir::{r#type::IntegerType, Location, Module},
//        Context,
        Context,
//    };
    };
//

//    #[test]
    #[test]
//    fn enum_type_single_variant_no_i0() {
    fn enum_type_single_variant_no_i0() {
//        let (_, program) = load_cairo! {
        let (_, program) = load_cairo! {
//            enum MyEnum {
            enum MyEnum {
//                A: felt252,
                A: felt252,
//            }
            }
//

//            fn run_program(x: MyEnum) -> MyEnum {
            fn run_program(x: MyEnum) -> MyEnum {
//                x
                x
//            }
            }
//        };
        };
//

//        let context = Context::new();
        let context = Context::new();
//        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();
//

//        let module = Module::new(Location::unknown(&context));
        let module = Module::new(Location::unknown(&context));
//        let mut metadata = MetadataStorage::new();
        let mut metadata = MetadataStorage::new();
//

//        let i0_ty = IntegerType::new(&context, 0).into();
        let i0_ty = IntegerType::new(&context, 0).into();
//        program
        program
//            .type_declarations
            .type_declarations
//            .iter()
            .iter()
//            .map(|ty| (&ty.id, registry.get_type(&ty.id).unwrap()))
            .map(|ty| (&ty.id, registry.get_type(&ty.id).unwrap()))
//            .map(|(id, ty)| {
            .map(|(id, ty)| {
//                ty.build(&context, &module, &registry, &mut metadata, id)
                ty.build(&context, &module, &registry, &mut metadata, id)
//                    .unwrap()
                    .unwrap()
//            })
            })
//            .any(|width| width == i0_ty);
            .any(|width| width == i0_ty);
//    }
    }
//}
}
