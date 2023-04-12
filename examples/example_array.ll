; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_new<u32>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 32)
  %2 = insertvalue { i32, i32, ptr } { i32 0, i32 8, ptr undef }, ptr %1, 2
  ret { i32, i32, ptr } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %0, i32 %1) #0 {
  %3 = extractvalue { i32, i32, ptr } %0, 0
  %4 = extractvalue { i32, i32, ptr } %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue { i32, i32, ptr } %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue { i32, i32, ptr } %0, ptr %10, 2
  %12 = insertvalue { i32, i32, ptr } %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi { i32, i32, ptr } [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue { i32, i32, ptr } %14, 2
  %16 = getelementptr i32, ptr %15, i32 %3
  store i32 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue { i32, i32, ptr } %14, i32 %17, 0
  ret { i32, i32, ptr } %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"array_len<u32>"({ i32, i32, ptr } %0) #0 {
  %2 = extractvalue { i32, i32, ptr } %0, 0
  ret i32 %2
}

; Function Attrs: norecurse nounwind
define internal void @print_u32(i32 %0) #1 {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i32 %0)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_Array<u32>"({ i32, i32, ptr } %0) #1 {
  %2 = extractvalue { i32, i32, ptr } %0, 2
  %3 = extractvalue { i32, i32, ptr } %0, 0
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp ult i32 %5, %3
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = phi i32 [ %5, %4 ]
  %9 = getelementptr i32, ptr %2, i32 %8
  %10 = load i32, ptr %9, align 4
  call void @print_u32(i32 %10)
  %11 = add i32 %8, 1
  br label %4

12:                                               ; preds = %4
  ret void
}

define void @main() {
  %1 = call { i32, i32, ptr } @"example_array::example_array::main"()
  call void @"print_Array<u32>"({ i32, i32, ptr } %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define { i32, i32, ptr } @"example_array::example_array::main"() {
  br label %1

1:                                                ; preds = %0
  %2 = call { i32, i32, ptr } @"array_new<u32>"()
  %3 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %2, i32 1)
  %4 = call {} @"struct_construct<Unit>"()
  %5 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %3, i32 2)
  %6 = call {} @"struct_construct<Unit>"()
  %7 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %5, i32 3)
  %8 = call {} @"struct_construct<Unit>"()
  %9 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %7, i32 4)
  %10 = call {} @"struct_construct<Unit>"()
  %11 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %9, i32 5)
  %12 = call {} @"struct_construct<Unit>"()
  %13 = call i32 @"array_len<u32>"({ i32, i32, ptr } %11)
  %14 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %11, i32 %13)
  %15 = call {} @"struct_construct<Unit>"()
  ret { i32, i32, ptr } %14
}

define void @"_mlir_ciface_example_array::example_array::main"(ptr %0) {
  %2 = call { i32, i32, ptr } @"example_array::example_array::main"()
  store { i32, i32, ptr } %2, ptr %0, align 8
  ret void
}

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
