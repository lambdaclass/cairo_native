; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal i16 @"upcast<u8, u16>"(i8 %0) #0 {
  %2 = zext i8 %0 to i16
  ret i16 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"upcast<u8, u32>"(i8 %0) #0 {
  %2 = zext i8 %0 to i32
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u8, u64>"(i8 %0) #0 {
  %2 = zext i8 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u8, u128>"(i8 %0) #0 {
  %2 = zext i8 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"upcast<u16, u32>"(i16 %0) #0 {
  %2 = zext i16 %0 to i32
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u16, u64>"(i16 %0) #0 {
  %2 = zext i16 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u16, u128>"(i16 %0) #0 {
  %2 = zext i16 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i64 @"upcast<u32, u64>"(i32 %0) #0 {
  %2 = zext i32 %0 to i64
  ret i64 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u32, u128>"(i32 %0) #0 {
  %2 = zext i32 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i128 @"upcast<u64, u128>"(i64 %0) #0 {
  %2 = zext i64 %0 to i128
  ret i128 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

; Function Attrs: norecurse nounwind
define internal void @print_Unit({} %0) #1 {
  ret void
}

define void @main() {
  %1 = call {} @"casts::casts::main"()
  call void @print_Unit({} %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define {} @"casts::casts::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call i16 @"upcast<u8, u16>"(i8 0)
  %3 = call i32 @"upcast<u8, u32>"(i8 0)
  %4 = call i64 @"upcast<u8, u64>"(i8 0)
  %5 = call i128 @"upcast<u8, u128>"(i8 0)
  %6 = call i32 @"upcast<u16, u32>"(i16 0)
  %7 = call i64 @"upcast<u16, u64>"(i16 0)
  %8 = call i128 @"upcast<u16, u128>"(i16 0)
  %9 = call i64 @"upcast<u32, u64>"(i32 0)
  %10 = call i128 @"upcast<u32, u128>"(i32 0)
  %11 = call i128 @"upcast<u64, u128>"(i64 0)
  %12 = call {} @"struct_construct<Unit>"()
  ret {} %12
}

define void @"_mlir_ciface_casts::casts::main"(ptr %0) {
  %2 = call {} @"casts::casts::main"()
  store {} %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
