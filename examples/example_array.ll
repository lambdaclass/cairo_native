; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

declare i32 @dprintf(i32, ptr, ...)

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_new<u32>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 32)
  %2 = insertvalue <{ i32, i32, ptr }> <{ i32 0, i32 8, ptr undef }>, ptr %1, 2
  ret <{ i32, i32, ptr }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %0, i32 %1) #0 {
  %3 = extractvalue <{ i32, i32, ptr }> %0, 0
  %4 = extractvalue <{ i32, i32, ptr }> %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue <{ i32, i32, ptr }> %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue <{ i32, i32, ptr }> %0, ptr %10, 2
  %12 = insertvalue <{ i32, i32, ptr }> %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi <{ i32, i32, ptr }> [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue <{ i32, i32, ptr }> %14, 2
  %16 = getelementptr i32, ptr %15, i32 %3
  store i32 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue <{ i32, i32, ptr }> %14, i32 %17, 0
  ret <{ i32, i32, ptr }> %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, i32 }>, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{}> @"struct_construct<Unit>"() #0 {
  ret <{}> undef
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %0) #0 {
  %2 = alloca <{ i16, [4 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [4 x i8] }>, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{}> }>, ptr %2, i32 0, i32 1
  store <{}> %0, ptr %4, align 1
  %5 = load <{ i16, [4 x i8] }>, ptr %2, align 1
  ret <{ i16, [4 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"array_len<u32>"(<{ i32, i32, ptr }> %0) #0 {
  %2 = extractvalue <{ i32, i32, ptr }> %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %0) #0 {
  %2 = extractvalue <{ i32 }> %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, i32, i32, i32 }> @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4) #0 {
  %6 = insertvalue <{ i32, i32, i32, i32, i32 }> undef, i32 %0, 0
  %7 = insertvalue <{ i32, i32, i32, i32, i32 }> %6, i32 %1, 1
  %8 = insertvalue <{ i32, i32, i32, i32, i32 }> %7, i32 %2, 2
  %9 = insertvalue <{ i32, i32, i32, i32, i32 }> %8, i32 %3, 3
  %10 = insertvalue <{ i32, i32, i32, i32, i32 }> %9, i32 %4, 4
  ret <{ i32, i32, i32, i32, i32 }> %10
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ <{ i32, i32, i32, i32, i32 }> }> @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(<{ i32, i32, i32, i32, i32 }> %0) #0 {
  %2 = insertvalue <{ <{ i32, i32, i32, i32, i32 }> }> undef, <{ i32, i32, i32, i32, i32 }> %0, 0
  ret <{ <{ i32, i32, i32, i32, i32 }> }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(<{ <{ i32, i32, i32, i32, i32 }> }> %0) #0 {
  %2 = alloca <{ i16, [20 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [20 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ <{ i32, i32, i32, i32, i32 }> }> }>, ptr %2, i32 0, i32 1
  store <{ <{ i32, i32, i32, i32, i32 }> }> %0, ptr %4, align 1
  %5 = load <{ i16, [20 x i8] }>, ptr %2, align 1
  ret <{ i16, [20 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<Box<u32>>>"(<{ i32 }> %0) #0 {
  %2 = extractvalue <{ i32 }> %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32 }> @"struct_construct<Tuple<u32>>"(i32 %0) #0 {
  %2 = insertvalue <{ i32 }> undef, i32 %0, 0
  ret <{ i32 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(<{ i32 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %2, i32 0, i32 1
  store <{ i32 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32 }> @"struct_construct<Tuple<Box<u32>>>"(i32 %0) #0 {
  %2 = insertvalue <{ i32 }> undef, i32 %0, 0
  ret <{ i32 }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(<{ i32 }> %0) #0 {
  %2 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  %3 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %2, i32 0, i32 1
  store <{ i32 }> %0, ptr %4, align 1
  %5 = load <{ i16, [16 x i8] }>, ptr %2, align 1
  ret <{ i16, [16 x i8] }> %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_new<felt252>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 256)
  %2 = insertvalue <{ i32, i32, ptr }> <{ i32 0, i32 8, ptr undef }>, ptr %1, 2
  ret <{ i32, i32, ptr }> %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %0, i256 %1) #0 {
  %3 = extractvalue <{ i32, i32, ptr }> %0, 0
  %4 = extractvalue <{ i32, i32, ptr }> %0, 1
  %5 = icmp ult i32 %3, %4
  br i1 %5, label %13, label %6

6:                                                ; preds = %2
  %7 = mul i32 %4, 2
  %8 = zext i32 %7 to i64
  %9 = extractvalue <{ i32, i32, ptr }> %0, 2
  %10 = call ptr @realloc(ptr %9, i64 %8)
  %11 = insertvalue <{ i32, i32, ptr }> %0, ptr %10, 2
  %12 = insertvalue <{ i32, i32, ptr }> %11, i32 %7, 1
  br label %13

13:                                               ; preds = %6, %2
  %14 = phi <{ i32, i32, ptr }> [ %12, %6 ], [ %0, %2 ]
  %15 = extractvalue <{ i32, i32, ptr }> %14, 2
  %16 = getelementptr i256, ptr %15, i32 %3
  store i256 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue <{ i32, i32, ptr }> %14, i32 %17, 0
  ret <{ i32, i32, ptr }> %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(<{ i32, i32, ptr }> %0) #0 {
  call void @llvm.trap()
  unreachable
}

; Function Attrs: norecurse nounwind
define internal void @print_u32(i32 %0) #1 {
  %2 = alloca i8, i64 4, align 1
  store [4 x i8] c"%X\0A\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2, i32 %0)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_Tuple<u32, u32, u32, u32, u32>"(<{ i32, i32, i32, i32, i32 }> %0) #1 {
  %2 = extractvalue <{ i32, i32, i32, i32, i32 }> %0, 0
  call void @print_u32(i32 %2)
  %3 = extractvalue <{ i32, i32, i32, i32, i32 }> %0, 1
  call void @print_u32(i32 %3)
  %4 = extractvalue <{ i32, i32, i32, i32, i32 }> %0, 2
  call void @print_u32(i32 %4)
  %5 = extractvalue <{ i32, i32, i32, i32, i32 }> %0, 3
  call void @print_u32(i32 %5)
  %6 = extractvalue <{ i32, i32, i32, i32, i32 }> %0, 4
  call void @print_u32(i32 %6)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(<{ <{ i32, i32, i32, i32, i32 }> }> %0) #1 {
  %2 = extractvalue <{ <{ i32, i32, i32, i32, i32 }> }> %0, 0
  call void @"print_Tuple<u32, u32, u32, u32, u32>"(<{ i32, i32, i32, i32, i32 }> %2)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @print_felt252(i256 %0) #1 {
  %2 = ashr i256 %0, 224
  %3 = trunc i256 %2 to i32
  %4 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %4, align 1
  %5 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %4, i32 %3)
  %6 = ashr i256 %0, 192
  %7 = trunc i256 %6 to i32
  %8 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %8, align 1
  %9 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %8, i32 %7)
  %10 = ashr i256 %0, 160
  %11 = trunc i256 %10 to i32
  %12 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %12, align 1
  %13 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %12, i32 %11)
  %14 = ashr i256 %0, 128
  %15 = trunc i256 %14 to i32
  %16 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %16, align 1
  %17 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %16, i32 %15)
  %18 = ashr i256 %0, 96
  %19 = trunc i256 %18 to i32
  %20 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %20, align 1
  %21 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %20, i32 %19)
  %22 = ashr i256 %0, 64
  %23 = trunc i256 %22 to i32
  %24 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %24, align 1
  %25 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %24, i32 %23)
  %26 = ashr i256 %0, 32
  %27 = trunc i256 %26 to i32
  %28 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %28, align 1
  %29 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %28, i32 %27)
  %30 = trunc i256 %0 to i32
  %31 = alloca i8, i64 5, align 1
  store [5 x i8] c"%08X\00", ptr %31, align 1
  %32 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %31, i32 %30)
  %33 = alloca i8, i64 2, align 1
  store [2 x i8] c"\0A\00", ptr %33, align 1
  %34 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %33)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_Array<felt252>"(<{ i32, i32, ptr }> %0) #1 {
  %2 = extractvalue <{ i32, i32, ptr }> %0, 2
  %3 = extractvalue <{ i32, i32, ptr }> %0, 0
  br label %4

4:                                                ; preds = %7, %1
  %5 = phi i32 [ %11, %7 ], [ 0, %1 ]
  %6 = icmp ult i32 %5, %3
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = phi i32 [ %5, %4 ]
  %9 = getelementptr i256, ptr %2, i32 %8
  %10 = load i256, ptr %9, align 4
  call void @print_felt252(i256 %10)
  %11 = add i32 %8, 1
  br label %4

12:                                               ; preds = %4
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(<{ i16, [20 x i8] }> %0) #1 {
  %2 = alloca <{ i16, [20 x i8] }>, i64 1, align 8
  store <{ i16, [20 x i8] }> %0, ptr %2, align 1
  %3 = getelementptr inbounds <{ i16, [20 x i8] }>, ptr %2, i32 0, i32 0
  %4 = load i16, ptr %3, align 2
  switch i16 %4, label %11 [
    i16 0, label %5
    i16 1, label %8
  ]

5:                                                ; preds = %1
  %6 = getelementptr inbounds <{ i16, <{ <{ i32, i32, i32, i32, i32 }> }> }>, ptr %2, i32 0, i32 1
  %7 = load <{ <{ i32, i32, i32, i32, i32 }> }>, ptr %6, align 1
  call void @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(<{ <{ i32, i32, i32, i32, i32 }> }> %7)
  ret void

8:                                                ; preds = %1
  %9 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %2, i32 0, i32 1
  %10 = load <{ i32, i32, ptr }>, ptr %9, align 1
  call void @"print_Array<felt252>"(<{ i32, i32, ptr }> %10)
  ret void

11:                                               ; preds = %1
  ret void
}

define void @main() {
  %1 = call <{ i16, [20 x i8] }> @"example_array::example_array::main"()
  call void @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(<{ i16, [20 x i8] }> %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define <{ i16, [20 x i8] }> @"example_array::example_array::main"() {
  br label %46

1:                                                ; preds = %46
  %2 = extractvalue <{ i32, i32, ptr }> %50, 2
  %3 = load i32, ptr %2, align 4
  %4 = sub i32 %51, 1
  %5 = getelementptr i32, ptr %2, i32 1
  %6 = zext i32 %4 to i64
  %7 = mul i64 %6, 4
  %8 = call ptr @memmove(ptr %2, ptr %5, i64 %7)
  %9 = insertvalue <{ i32, i32, ptr }> %50, i32 %4, 0
  %10 = insertvalue <{ i32, i32, ptr }> %9, ptr %8, 2
  br label %53

11:                                               ; preds = %61
  %12 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %68, i32 0, i32 1
  %13 = load <{ i32 }>, ptr %12, align 1
  br label %71

14:                                               ; preds = %61
  %15 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %68, i32 0, i32 1
  %16 = load <{ i32, i32, ptr }>, ptr %15, align 1
  br label %131

17:                                               ; preds = %61
  unreachable

18:                                               ; preds = %71
  %19 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %76, i32 0, i32 1
  %20 = load <{ i32 }>, ptr %19, align 1
  br label %79

21:                                               ; preds = %71
  %22 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %76, i32 0, i32 1
  %23 = load <{ i32, i32, ptr }>, ptr %22, align 1
  br label %128

24:                                               ; preds = %71
  unreachable

25:                                               ; preds = %79
  %26 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %85, i32 0, i32 1
  %27 = load <{ i32 }>, ptr %26, align 1
  br label %88

28:                                               ; preds = %79
  %29 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %85, i32 0, i32 1
  %30 = load <{ i32, i32, ptr }>, ptr %29, align 1
  br label %125

31:                                               ; preds = %79
  unreachable

32:                                               ; preds = %88
  %33 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %95, i32 0, i32 1
  %34 = load <{ i32 }>, ptr %33, align 1
  br label %98

35:                                               ; preds = %88
  %36 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %95, i32 0, i32 1
  %37 = load <{ i32, i32, ptr }>, ptr %36, align 1
  br label %122

38:                                               ; preds = %88
  unreachable

39:                                               ; preds = %98
  %40 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %106, i32 0, i32 1
  %41 = load <{ i32 }>, ptr %40, align 1
  br label %109

42:                                               ; preds = %98
  %43 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %106, i32 0, i32 1
  %44 = load <{ i32, i32, ptr }>, ptr %43, align 1
  br label %119

45:                                               ; preds = %98
  unreachable

46:                                               ; preds = %0
  %47 = call <{ i32, i32, ptr }> @"array_new<u32>"()
  %48 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %47, i32 1)
  %49 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %48, i32 2)
  %50 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %49, i32 3)
  %51 = extractvalue <{ i32, i32, ptr }> %50, 0
  %52 = icmp uge i32 %51, 1
  br i1 %52, label %1, label %57

53:                                               ; preds = %1
  %54 = phi <{ i32, i32, ptr }> [ %10, %1 ]
  %55 = phi i32 [ %3, %1 ]
  %56 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %55)
  br label %61

57:                                               ; preds = %46
  %58 = phi <{ i32, i32, ptr }> [ %50, %46 ]
  %59 = call <{}> @"struct_construct<Unit>"()
  %60 = call <{ i16, [4 x i8] }> @"enum_init<core::option::Option::<core::integer::u32>, 1>"(<{}> %59)
  br label %61

61:                                               ; preds = %53, %57
  %62 = phi <{ i32, i32, ptr }> [ %58, %57 ], [ %54, %53 ]
  %63 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %62, i32 7)
  %64 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %63, i32 5)
  %65 = call i32 @"array_len<u32>"(<{ i32, i32, ptr }> %64)
  %66 = call <{ i32, i32, ptr }> @"array_append<u32>"(<{ i32, i32, ptr }> %64, i32 %65)
  %67 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %66, i32 0)
  %68 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %67, ptr %68, align 1
  %69 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %68, i32 0, i32 0
  %70 = load i16, ptr %69, align 2
  switch i16 %70, label %17 [
    i16 0, label %11
    i16 1, label %14
  ]

71:                                               ; preds = %11
  %72 = phi <{ i32, i32, ptr }> [ %66, %11 ]
  %73 = phi <{ i32 }> [ %13, %11 ]
  %74 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %73)
  %75 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %72, i32 1)
  %76 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %75, ptr %76, align 1
  %77 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %76, i32 0, i32 0
  %78 = load i16, ptr %77, align 2
  switch i16 %78, label %24 [
    i16 0, label %18
    i16 1, label %21
  ]

79:                                               ; preds = %18
  %80 = phi i32 [ %74, %18 ]
  %81 = phi <{ i32, i32, ptr }> [ %72, %18 ]
  %82 = phi <{ i32 }> [ %20, %18 ]
  %83 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %82)
  %84 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %81, i32 2)
  %85 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %84, ptr %85, align 1
  %86 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %85, i32 0, i32 0
  %87 = load i16, ptr %86, align 2
  switch i16 %87, label %31 [
    i16 0, label %25
    i16 1, label %28
  ]

88:                                               ; preds = %25
  %89 = phi i32 [ %80, %25 ]
  %90 = phi i32 [ %83, %25 ]
  %91 = phi <{ i32, i32, ptr }> [ %81, %25 ]
  %92 = phi <{ i32 }> [ %27, %25 ]
  %93 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %92)
  %94 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %91, i32 3)
  %95 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %94, ptr %95, align 1
  %96 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %95, i32 0, i32 0
  %97 = load i16, ptr %96, align 2
  switch i16 %97, label %38 [
    i16 0, label %32
    i16 1, label %35
  ]

98:                                               ; preds = %32
  %99 = phi i32 [ %89, %32 ]
  %100 = phi i32 [ %90, %32 ]
  %101 = phi i32 [ %93, %32 ]
  %102 = phi <{ i32, i32, ptr }> [ %91, %32 ]
  %103 = phi <{ i32 }> [ %34, %32 ]
  %104 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %103)
  %105 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %102, i32 4)
  %106 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %105, ptr %106, align 1
  %107 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %106, i32 0, i32 0
  %108 = load i16, ptr %107, align 2
  switch i16 %108, label %45 [
    i16 0, label %39
    i16 1, label %42
  ]

109:                                              ; preds = %39
  %110 = phi i32 [ %99, %39 ]
  %111 = phi i32 [ %100, %39 ]
  %112 = phi i32 [ %101, %39 ]
  %113 = phi i32 [ %104, %39 ]
  %114 = phi <{ i32 }> [ %41, %39 ]
  %115 = call i32 @"struct_deconstruct<Tuple<u32>>"(<{ i32 }> %114)
  %116 = call <{ i32, i32, i32, i32, i32 }> @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(i32 %110, i32 %111, i32 %112, i32 %113, i32 %115)
  %117 = call <{ <{ i32, i32, i32, i32, i32 }> }> @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(<{ i32, i32, i32, i32, i32 }> %116)
  %118 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(<{ <{ i32, i32, i32, i32, i32 }> }> %117)
  ret <{ i16, [20 x i8] }> %118

119:                                              ; preds = %42
  %120 = phi <{ i32, i32, ptr }> [ %44, %42 ]
  %121 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %120)
  ret <{ i16, [20 x i8] }> %121

122:                                              ; preds = %35
  %123 = phi <{ i32, i32, ptr }> [ %37, %35 ]
  %124 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %123)
  ret <{ i16, [20 x i8] }> %124

125:                                              ; preds = %28
  %126 = phi <{ i32, i32, ptr }> [ %30, %28 ]
  %127 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %126)
  ret <{ i16, [20 x i8] }> %127

128:                                              ; preds = %21
  %129 = phi <{ i32, i32, ptr }> [ %23, %21 ]
  %130 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %129)
  ret <{ i16, [20 x i8] }> %130

131:                                              ; preds = %14
  %132 = phi <{ i32, i32, ptr }> [ %16, %14 ]
  %133 = call <{ i16, [20 x i8] }> @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(<{ i32, i32, ptr }> %132)
  ret <{ i16, [20 x i8] }> %133
}

define void @"_mlir_ciface_example_array::example_array::main"(ptr %0) {
  %2 = call <{ i16, [20 x i8] }> @"example_array::example_array::main"()
  store <{ i16, [20 x i8] }> %2, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %0, i32 %1) {
  br label %10

3:                                                ; preds = %10
  %4 = getelementptr inbounds <{ i16, <{ i32 }> }>, ptr %14, i32 0, i32 1
  %5 = load <{ i32 }>, ptr %4, align 1
  br label %17

6:                                                ; preds = %10
  %7 = getelementptr inbounds <{ i16, <{ i32, i32, ptr }> }>, ptr %14, i32 0, i32 1
  %8 = load <{ i32, i32, ptr }>, ptr %7, align 1
  br label %22

9:                                                ; preds = %10
  unreachable

10:                                               ; preds = %2
  %11 = phi <{ i32, i32, ptr }> [ %0, %2 ]
  %12 = phi i32 [ %1, %2 ]
  %13 = call <{ i16, [16 x i8] }> @"core::array::array_at::<core::integer::u32>"(<{ i32, i32, ptr }> %11, i32 %12)
  %14 = alloca <{ i16, [16 x i8] }>, i64 1, align 8
  store <{ i16, [16 x i8] }> %13, ptr %14, align 1
  %15 = getelementptr inbounds <{ i16, [16 x i8] }>, ptr %14, i32 0, i32 0
  %16 = load i16, ptr %15, align 2
  switch i16 %16, label %9 [
    i16 0, label %3
    i16 1, label %6
  ]

17:                                               ; preds = %3
  %18 = phi <{ i32 }> [ %5, %3 ]
  %19 = call i32 @"struct_deconstruct<Tuple<Box<u32>>>"(<{ i32 }> %18)
  %20 = call <{ i32 }> @"struct_construct<Tuple<u32>>"(i32 %19)
  %21 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(<{ i32 }> %20)
  ret <{ i16, [16 x i8] }> %21

22:                                               ; preds = %6
  %23 = phi <{ i32, i32, ptr }> [ %8, %6 ]
  %24 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(<{ i32, i32, ptr }> %23)
  ret <{ i16, [16 x i8] }> %24
}

define void @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(ptr %0, <{ i32, i32, ptr }> %1, i32 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::array::ArrayIndex::<core::integer::u32>::index"(<{ i32, i32, ptr }> %1, i32 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

define <{ i16, [16 x i8] }> @"core::array::array_at::<core::integer::u32>"(<{ i32, i32, ptr }> %0, i32 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = extractvalue <{ i32, i32, ptr }> %8, 2
  %5 = getelementptr i32, ptr %4, i32 %9
  %6 = load i32, ptr %5, align 4
  br label %12

7:                                                ; preds = %2
  %8 = phi <{ i32, i32, ptr }> [ %0, %2 ]
  %9 = phi i32 [ %1, %2 ]
  %10 = extractvalue <{ i32, i32, ptr }> %8, 0
  %11 = icmp ult i32 %9, %10
  br i1 %11, label %3, label %16

12:                                               ; preds = %3
  %13 = phi i32 [ %6, %3 ]
  %14 = call <{ i32 }> @"struct_construct<Tuple<Box<u32>>>"(i32 %13)
  %15 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(<{ i32 }> %14)
  ret <{ i16, [16 x i8] }> %15

16:                                               ; preds = %7
  %17 = call <{ i32, i32, ptr }> @"array_new<felt252>"()
  %18 = call <{ i32, i32, ptr }> @"array_append<felt252>"(<{ i32, i32, ptr }> %17, i256 1637570914057682275393755530660268060279989363)
  %19 = call <{ i16, [16 x i8] }> @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(<{ i32, i32, ptr }> %18)
  ret <{ i16, [16 x i8] }> %19
}

define void @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(ptr %0, <{ i32, i32, ptr }> %1, i32 %2) {
  %4 = call <{ i16, [16 x i8] }> @"core::array::array_at::<core::integer::u32>"(<{ i32, i32, ptr }> %1, i32 %2)
  store <{ i16, [16 x i8] }> %4, ptr %0, align 1
  ret void
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }
attributes #2 = { cold noreturn nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
