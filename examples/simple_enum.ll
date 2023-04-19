; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 0>"(i8 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i8 }>, ptr %2, i32 0, i32 1
  store i8 %0, ptr %4, align 1
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 1>"(i16 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 2>"(i32 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 2, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(i64 %0) #0 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  store i16 3, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i64 }>, ptr %2, i32 0, i32 1
  store i64 %0, ptr %4, align 4
  %5 = load <{ i16, [8 x i8] }>, ptr %2, align 1
  ret <{ i16, [8 x i8] }> %5
}

; Function Attrs: norecurse nounwind
define internal void @print_u8(i8 %0) #1 {
  %2 = zext i8 %0 to i32
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i32 %2)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @print_u16(i16 %0) #1 {
  %2 = zext i16 %0 to i32
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i32 %2)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @print_u32(i32 %0) #1 {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i32 %0)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @print_u64(i64 %0) #1 {
  %2 = alloca i8, i64 5, align 1
  store [5 x i8] c"%lX\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i64 %0)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_simple_enum::simple_enum::MyEnum"(<{ i16, [8 x i8] }> %0) #1 {
  %2 = alloca <{ i16, [8 x i8] }>, i64 1, align 8
  store <{ i16, [8 x i8] }> %0, ptr %2, align 1
  %3 = getelementptr inbounds <{ i16, [8 x i8] }>, ptr %2, i32 0, i32 0
  %4 = load i16, ptr %3, align 2
  %5 = zext i16 %4 to i32
  %6 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %6, align 1
  %7 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %6, i32 %5)
  switch i16 %4, label %20 [
    i16 0, label %8
    i16 1, label %11
    i16 2, label %14
    i16 3, label %17
  ]

8:                                                ; preds = %1
  %9 = getelementptr inbounds <{ i16, i8 }>, ptr %2, i32 0, i32 1
  %10 = load i8, ptr %9, align 1
  call void @print_u8(i8 %10)
  ret void

11:                                               ; preds = %1
  %12 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  %13 = load i16, ptr %12, align 2
  call void @print_u16(i16 %13)
  ret void

14:                                               ; preds = %1
  %15 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  %16 = load i32, ptr %15, align 4
  call void @print_u32(i32 %16)
  ret void

17:                                               ; preds = %1
  %18 = getelementptr inbounds <{ i16, i64 }>, ptr %2, i32 0, i32 1
  %19 = load i64, ptr %18, align 4
  call void @print_u64(i64 %19)
  ret void

20:                                               ; preds = %1
  ret void
}

define void @main() {
  %1 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::main"()
  call void @"print_simple_enum::simple_enum::MyEnum"(<{ i16, [8 x i8] }> %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_a"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 0>"(i8 4)
  ret <{ i16, [8 x i8] }> %2
}

define void @"_mlir_ciface_simple_enum::simple_enum::my_enum_a"(ptr %0) {
  %2 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_a"()
  store <{ i16, [8 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_b"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 1>"(i16 8)
  ret <{ i16, [8 x i8] }> %2
}

define void @"_mlir_ciface_simple_enum::simple_enum::my_enum_b"(ptr %0) {
  %2 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_b"()
  store <{ i16, [8 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_c"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 2>"(i32 16)
  ret <{ i16, [8 x i8] }> %2
}

define void @"_mlir_ciface_simple_enum::simple_enum::my_enum_c"(ptr %0) {
  %2 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_c"()
  store <{ i16, [8 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_d"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(i64 16)
  ret <{ i16, [8 x i8] }> %2
}

define void @"_mlir_ciface_simple_enum::simple_enum::my_enum_d"(ptr %0) {
  %2 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::my_enum_d"()
  store <{ i16, [8 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [8 x i8] }> @"simple_enum::simple_enum::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [8 x i8] }> @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(i64 16)
  ret <{ i16, [8 x i8] }> %2
}

define void @"_mlir_ciface_simple_enum::simple_enum::main"(ptr %0) {
  %2 = call <{ i16, [8 x i8] }> @"simple_enum::simple_enum::main"()
  store <{ i16, [8 x i8] }> %2, ptr %0, align 1
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
