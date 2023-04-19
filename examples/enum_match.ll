; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<enum_match::enum_match::MyEnum, 1>"(i16 %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %2, i32 0, i32 1
  store i16 %0, ptr %4, align 2
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: norecurse nounwind
define internal void @print_u16(i16 %0) #1 {
  %2 = zext i16 %0 to i32
  %3 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %3, align 1
  %4 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %3, i32 %2)
  ret void
}

define void @main() {
  %1 = call i16 @"enum_match::enum_match::main"()
  call void @print_u16(i16 %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define i16 @"enum_match::enum_match::get_my_enum_b"(<{ i16, [4 x i8] }> %0) {
  br label %8

2:                                                ; preds = %8
  br label %13

3:                                                ; preds = %8
  %4 = getelementptr inbounds <{ i16, i16 }>, ptr %10, i32 0, i32 1
  %5 = load i16, ptr %4, align 2
  br label %14

6:                                                ; preds = %8
  br label %16

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %1
  %9 = phi <{ i16, [4 x i8] }> [ %0, %1 ]
  %10 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  store <{ i16, [4 x i8] }> %9, ptr %10, align 1
  %11 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %10, i32 0, i32 0
  %12 = load i16, ptr %11, align 2
  switch i16 %12, label %7 [
    i16 0, label %2
    i16 1, label %3
    i16 2, label %6
  ]

13:                                               ; preds = %2
  br label %17

14:                                               ; preds = %3
  %15 = phi i16 [ %5, %3 ]
  br label %17

16:                                               ; preds = %6
  br label %17

17:                                               ; preds = %13, %14, %16
  %18 = phi i16 [ 0, %16 ], [ %15, %14 ], [ 1, %13 ]
  ret i16 %18
}

define i16 @"_mlir_ciface_enum_match::enum_match::get_my_enum_b"(<{ i16, [4 x i8] }> %0) {
  %2 = call i16 @"enum_match::enum_match::get_my_enum_b"(<{ i16, [4 x i8] }> %0)
  ret i16 %2
}

define i16 @"enum_match::enum_match::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call <{ i16, [4 x i8] }> @"enum_init<enum_match::enum_match::MyEnum, 1>"(i16 16)
  %3 = call i16 @"enum_match::enum_match::get_my_enum_b"(<{ i16, [4 x i8] }> %2)
  ret i16 %3
}

define i16 @"_mlir_ciface_enum_match::enum_match::main"() {
  %1 = call i16 @"enum_match::enum_match::main"()
  ret i16 %1
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
