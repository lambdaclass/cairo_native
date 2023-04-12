; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal i256 @u8_to_felt252(i8 %0) {
  %2 = zext i8 %0 to i256
  ret i256 %2
}

define internal i256 @u16_to_felt252(i16 %0) {
  %2 = zext i16 %0 to i256
  ret i256 %2
}

define internal i256 @u32_to_felt252(i32 %0) {
  %2 = zext i32 %0 to i256
  ret i256 %2
}

define internal i256 @u64_to_felt252(i64 %0) {
  %2 = zext i64 %0 to i256
  ret i256 %2
}

define internal i256 @u128_to_felt252(i128 %0) {
  %2 = zext i128 %0 to i256
  ret i256 %2
}

define internal i16 @u8_wide_mul(i8 %0, i8 %1) {
  %3 = zext i8 %0 to i16
  %4 = zext i8 %1 to i16
  %5 = mul i16 %3, %4
  ret i16 %5
}

define internal i32 @u16_wide_mul(i16 %0, i16 %1) {
  %3 = zext i16 %0 to i32
  %4 = zext i16 %1 to i32
  %5 = mul i32 %3, %4
  ret i32 %5
}

define internal i64 @u32_wide_mul(i32 %0, i32 %1) {
  %3 = zext i32 %0 to i64
  %4 = zext i32 %1 to i64
  %5 = mul i64 %3, %4
  ret i64 %5
}

define internal i128 @u64_wide_mul(i64 %0, i64 %1) {
  %3 = zext i64 %0 to i128
  %4 = zext i64 %1 to i128
  %5 = mul i128 %3, %4
  ret i128 %5
}

define internal { i128, i128 } @u128_wide_mul(i128 %0, i128 %1) {
  %3 = zext i128 %0 to i256
  %4 = zext i128 %1 to i256
  %5 = mul i256 %3, %4
  %6 = trunc i256 %5 to i128
  %7 = lshr i256 %5, 128
  %8 = trunc i256 %7 to i128
  %9 = insertvalue { i128, i128 } undef, i128 %8, 0
  %10 = insertvalue { i128, i128 } %9, i128 %6, 1
  ret { i128, i128 } %10
}

define internal {} @"struct_construct<Unit>"() {
  ret {} undef
}

define {} @"uint::uint::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call i256 @u8_to_felt252(i8 0)
  %3 = call i256 @u16_to_felt252(i16 0)
  %4 = call i256 @u32_to_felt252(i32 0)
  %5 = call i256 @u64_to_felt252(i64 0)
  %6 = call i256 @u128_to_felt252(i128 0)
  %7 = call i16 @u8_wide_mul(i8 0, i8 0)
  %8 = call i32 @u16_wide_mul(i16 0, i16 0)
  %9 = call i64 @u32_wide_mul(i32 0, i32 0)
  %10 = call i128 @u64_wide_mul(i64 0, i64 0)
  %11 = call { i128, i128 } @u128_wide_mul(i128 0, i128 0)
  %12 = extractvalue { i128, i128 } %11, 0
  %13 = extractvalue { i128, i128 } %11, 1
  %14 = call {} @"struct_construct<Unit>"()
  ret {} %14
}

define void @"_mlir_ciface_uint::uint::main"(ptr %0) {
  %2 = call {} @"uint::uint::main"()
  store {} %2, ptr %0, align 1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
