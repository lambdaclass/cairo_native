; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare ptr @realloc(ptr, i64)

declare ptr @memmove(ptr, ptr, i64)

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
define internal { i16, [4 x i8] } @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %0) #0 {
  %2 = alloca { i16, [4 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 1
  store i32 %0, ptr %4, align 4
  %5 = load { i16, [4 x i8] }, ptr %2, align 2
  ret { i16, [4 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal {} @"struct_construct<Unit>"() #0 {
  ret {} undef
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [4 x i8] } @"enum_init<core::option::Option::<core::integer::u32>, 1>"({} %0) #0 {
  %2 = alloca { i16, [4 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 0
  store i16 1, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [4 x i8] }, ptr %2, i32 0, i32 1
  store {} %0, ptr %4, align 1
  %5 = load { i16, [4 x i8] }, ptr %2, align 2
  ret { i16, [4 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"array_len<u32>"({ i32, i32, ptr } %0) #0 {
  %2 = extractvalue { i32, i32, ptr } %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %0) #0 {
  %2 = extractvalue { i32 } %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, i32, i32, i32 } @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4) #0 {
  %6 = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %0, 0
  %7 = insertvalue { i32, i32, i32, i32, i32 } %6, i32 %1, 1
  %8 = insertvalue { i32, i32, i32, i32, i32 } %7, i32 %2, 2
  %9 = insertvalue { i32, i32, i32, i32, i32 } %8, i32 %3, 3
  %10 = insertvalue { i32, i32, i32, i32, i32 } %9, i32 %4, 4
  ret { i32, i32, i32, i32, i32 } %10
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { { i32, i32, i32, i32, i32 } } @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"({ i32, i32, i32, i32, i32 } %0) #0 {
  %2 = insertvalue { { i32, i32, i32, i32, i32 } } undef, { i32, i32, i32, i32, i32 } %0, 0
  ret { { i32, i32, i32, i32, i32 } } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"({ { i32, i32, i32, i32, i32 } } %0) #0 {
  %2 = alloca { i16, [20 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [20 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [20 x i8] }, ptr %2, i32 0, i32 1
  store { { i32, i32, i32, i32, i32 } } %0, ptr %4, align 4
  %5 = load { i16, [20 x i8] }, ptr %2, align 2
  ret { i16, [20 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal i32 @"struct_deconstruct<Tuple<Box<u32>>>"({ i32 } %0) #0 {
  %2 = extractvalue { i32 } %0, 0
  ret i32 %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32 } @"struct_construct<Tuple<u32>>"(i32 %0) #0 {
  %2 = insertvalue { i32 } undef, i32 %0, 0
  ret { i32 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"({ i32 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i32 } %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
  call void @llvm.trap()
  unreachable
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32 } @"struct_construct<Tuple<Box<u32>>>"(i32 %0) #0 {
  %2 = insertvalue { i32 } undef, i32 %0, 0
  ret { i32 } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"({ i32 } %0) #0 {
  %2 = alloca { i16, [16 x i8] }, i64 1, align 8
  %3 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 0
  store i16 0, ptr %3, align 2
  %4 = getelementptr inbounds { i16, [16 x i8] }, ptr %2, i32 0, i32 1
  store { i32 } %0, ptr %4, align 4
  %5 = load { i16, [16 x i8] }, ptr %2, align 2
  ret { i16, [16 x i8] } %5
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_new<felt252>"() #0 {
  %1 = call ptr @realloc(ptr null, i64 256)
  %2 = insertvalue { i32, i32, ptr } { i32 0, i32 8, ptr undef }, ptr %1, 2
  ret { i32, i32, ptr } %2
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %0, i256 %1) #0 {
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
  %16 = getelementptr i256, ptr %15, i32 %3
  store i256 %1, ptr %16, align 4
  %17 = add i32 %3, 1
  %18 = insertvalue { i32, i32, ptr } %14, i32 %17, 0
  ret { i32, i32, ptr } %18
}

; Function Attrs: alwaysinline norecurse nounwind
define internal { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"({ i32, i32, ptr } %0) #0 {
  %2 = alloca i8, i64 13, align 1
  store [13 x i8] c"trap reached\00", ptr %2, align 1
  %3 = call i32 (i32, ptr, ...) @dprintf(i32 1, ptr %2)
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
define internal void @"print_Tuple<u32, u32, u32, u32, u32>"({ i32, i32, i32, i32, i32 } %0) #1 {
  %2 = extractvalue { i32, i32, i32, i32, i32 } %0, 0
  call void @print_u32(i32 %2)
  %3 = extractvalue { i32, i32, i32, i32, i32 } %0, 1
  call void @print_u32(i32 %3)
  %4 = extractvalue { i32, i32, i32, i32, i32 } %0, 2
  call void @print_u32(i32 %4)
  %5 = extractvalue { i32, i32, i32, i32, i32 } %0, 3
  call void @print_u32(i32 %5)
  %6 = extractvalue { i32, i32, i32, i32, i32 } %0, 4
  call void @print_u32(i32 %6)
  ret void
}

; Function Attrs: norecurse nounwind
define internal void @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"({ { i32, i32, i32, i32, i32 } } %0) #1 {
  %2 = extractvalue { { i32, i32, i32, i32, i32 } } %0, 0
  call void @"print_Tuple<u32, u32, u32, u32, u32>"({ i32, i32, i32, i32, i32 } %2)
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
define internal void @"print_Array<felt252>"({ i32, i32, ptr } %0) #1 {
  %2 = extractvalue { i32, i32, ptr } %0, 2
  %3 = extractvalue { i32, i32, ptr } %0, 0
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
define internal void @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"({ i16, [20 x i8] } %0) #1 {
  %2 = extractvalue { i16, [20 x i8] } %0, 0
  %3 = alloca [20 x i8], i64 1, align 1
  %4 = extractvalue { i16, [20 x i8] } %0, 1
  store [20 x i8] %4, ptr %3, align 1
  switch i16 %2, label %9 [
    i16 0, label %5
    i16 1, label %7
  ]

5:                                                ; preds = %1
  %6 = load { { i32, i32, i32, i32, i32 } }, ptr %3, align 4
  call void @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"({ { i32, i32, i32, i32, i32 } } %6)
  ret void

7:                                                ; preds = %1
  %8 = load { i32, i32, ptr }, ptr %3, align 8
  call void @"print_Array<felt252>"({ i32, i32, ptr } %8)
  ret void

9:                                                ; preds = %1
  ret void
}

define void @main() {
  %1 = call { i16, [20 x i8] } @"example_array::example_array::main"()
  call void @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"({ i16, [20 x i8] } %1)
  ret void
}

define void @_mlir_ciface_main() {
  call void @main()
  ret void
}

define { i16, [20 x i8] } @"example_array::example_array::main"() {
  br label %37

1:                                                ; preds = %37
  %2 = extractvalue { i32, i32, ptr } %41, 2
  %3 = getelementptr i32, ptr %2, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = sub i32 %42, 1
  %6 = getelementptr i32, ptr %2, i32 1
  %7 = zext i32 %5 to i64
  %8 = mul i64 %7, 4
  %9 = call ptr @memmove(ptr %2, ptr %6, i64 %8)
  %10 = insertvalue { i32, i32, ptr } %41, i32 %5, 0
  %11 = insertvalue { i32, i32, ptr } %10, ptr %9, 2
  br label %44

12:                                               ; preds = %52
  %13 = load { i32 }, ptr %61, align 4
  br label %62

14:                                               ; preds = %52
  %15 = load { i32, i32, ptr }, ptr %61, align 8
  br label %122

16:                                               ; preds = %52
  unreachable

17:                                               ; preds = %62
  %18 = load { i32 }, ptr %69, align 4
  br label %70

19:                                               ; preds = %62
  %20 = load { i32, i32, ptr }, ptr %69, align 8
  br label %119

21:                                               ; preds = %62
  unreachable

22:                                               ; preds = %70
  %23 = load { i32 }, ptr %78, align 4
  br label %79

24:                                               ; preds = %70
  %25 = load { i32, i32, ptr }, ptr %78, align 8
  br label %116

26:                                               ; preds = %70
  unreachable

27:                                               ; preds = %79
  %28 = load { i32 }, ptr %88, align 4
  br label %89

29:                                               ; preds = %79
  %30 = load { i32, i32, ptr }, ptr %88, align 8
  br label %113

31:                                               ; preds = %79
  unreachable

32:                                               ; preds = %89
  %33 = load { i32 }, ptr %99, align 4
  br label %100

34:                                               ; preds = %89
  %35 = load { i32, i32, ptr }, ptr %99, align 8
  br label %110

36:                                               ; preds = %89
  unreachable

37:                                               ; preds = %0
  %38 = call { i32, i32, ptr } @"array_new<u32>"()
  %39 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %38, i32 1)
  %40 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %39, i32 2)
  %41 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %40, i32 3)
  %42 = extractvalue { i32, i32, ptr } %41, 0
  %43 = icmp uge i32 %42, 1
  br i1 %43, label %1, label %48

44:                                               ; preds = %1
  %45 = phi { i32, i32, ptr } [ %11, %1 ]
  %46 = phi i32 [ %4, %1 ]
  %47 = call { i16, [4 x i8] } @"enum_init<core::option::Option::<core::integer::u32>, 0>"(i32 %46)
  br label %52

48:                                               ; preds = %37
  %49 = phi { i32, i32, ptr } [ %41, %37 ]
  %50 = call {} @"struct_construct<Unit>"()
  %51 = call { i16, [4 x i8] } @"enum_init<core::option::Option::<core::integer::u32>, 1>"({} %50)
  br label %52

52:                                               ; preds = %44, %48
  %53 = phi { i32, i32, ptr } [ %49, %48 ], [ %45, %44 ]
  %54 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %53, i32 7)
  %55 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %54, i32 5)
  %56 = call i32 @"array_len<u32>"({ i32, i32, ptr } %55)
  %57 = call { i32, i32, ptr } @"array_append<u32>"({ i32, i32, ptr } %55, i32 %56)
  %58 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %57, i32 0)
  %59 = extractvalue { i16, [16 x i8] } %58, 0
  %60 = extractvalue { i16, [16 x i8] } %58, 1
  %61 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %60, ptr %61, align 1
  switch i16 %59, label %16 [
    i16 0, label %12
    i16 1, label %14
  ]

62:                                               ; preds = %12
  %63 = phi { i32, i32, ptr } [ %57, %12 ]
  %64 = phi { i32 } [ %13, %12 ]
  %65 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %64)
  %66 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %63, i32 1)
  %67 = extractvalue { i16, [16 x i8] } %66, 0
  %68 = extractvalue { i16, [16 x i8] } %66, 1
  %69 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %68, ptr %69, align 1
  switch i16 %67, label %21 [
    i16 0, label %17
    i16 1, label %19
  ]

70:                                               ; preds = %17
  %71 = phi i32 [ %65, %17 ]
  %72 = phi { i32, i32, ptr } [ %63, %17 ]
  %73 = phi { i32 } [ %18, %17 ]
  %74 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %73)
  %75 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %72, i32 2)
  %76 = extractvalue { i16, [16 x i8] } %75, 0
  %77 = extractvalue { i16, [16 x i8] } %75, 1
  %78 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %77, ptr %78, align 1
  switch i16 %76, label %26 [
    i16 0, label %22
    i16 1, label %24
  ]

79:                                               ; preds = %22
  %80 = phi i32 [ %71, %22 ]
  %81 = phi i32 [ %74, %22 ]
  %82 = phi { i32, i32, ptr } [ %72, %22 ]
  %83 = phi { i32 } [ %23, %22 ]
  %84 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %83)
  %85 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %82, i32 3)
  %86 = extractvalue { i16, [16 x i8] } %85, 0
  %87 = extractvalue { i16, [16 x i8] } %85, 1
  %88 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %87, ptr %88, align 1
  switch i16 %86, label %31 [
    i16 0, label %27
    i16 1, label %29
  ]

89:                                               ; preds = %27
  %90 = phi i32 [ %80, %27 ]
  %91 = phi i32 [ %81, %27 ]
  %92 = phi i32 [ %84, %27 ]
  %93 = phi { i32, i32, ptr } [ %82, %27 ]
  %94 = phi { i32 } [ %28, %27 ]
  %95 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %94)
  %96 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %93, i32 4)
  %97 = extractvalue { i16, [16 x i8] } %96, 0
  %98 = extractvalue { i16, [16 x i8] } %96, 1
  %99 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %98, ptr %99, align 1
  switch i16 %97, label %36 [
    i16 0, label %32
    i16 1, label %34
  ]

100:                                              ; preds = %32
  %101 = phi i32 [ %90, %32 ]
  %102 = phi i32 [ %91, %32 ]
  %103 = phi i32 [ %92, %32 ]
  %104 = phi i32 [ %95, %32 ]
  %105 = phi { i32 } [ %33, %32 ]
  %106 = call i32 @"struct_deconstruct<Tuple<u32>>"({ i32 } %105)
  %107 = call { i32, i32, i32, i32, i32 } @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(i32 %101, i32 %102, i32 %103, i32 %104, i32 %106)
  %108 = call { { i32, i32, i32, i32, i32 } } @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"({ i32, i32, i32, i32, i32 } %107)
  %109 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"({ { i32, i32, i32, i32, i32 } } %108)
  ret { i16, [20 x i8] } %109

110:                                              ; preds = %34
  %111 = phi { i32, i32, ptr } [ %35, %34 ]
  %112 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %111)
  ret { i16, [20 x i8] } %112

113:                                              ; preds = %29
  %114 = phi { i32, i32, ptr } [ %30, %29 ]
  %115 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %114)
  ret { i16, [20 x i8] } %115

116:                                              ; preds = %24
  %117 = phi { i32, i32, ptr } [ %25, %24 ]
  %118 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %117)
  ret { i16, [20 x i8] } %118

119:                                              ; preds = %19
  %120 = phi { i32, i32, ptr } [ %20, %19 ]
  %121 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %120)
  ret { i16, [20 x i8] } %121

122:                                              ; preds = %14
  %123 = phi { i32, i32, ptr } [ %15, %14 ]
  %124 = call { i16, [20 x i8] } @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"({ i32, i32, ptr } %123)
  ret { i16, [20 x i8] } %124
}

define void @"_mlir_ciface_example_array::example_array::main"(ptr %0) {
  %2 = call { i16, [20 x i8] } @"example_array::example_array::main"()
  store { i16, [20 x i8] } %2, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %0, i32 %1) {
  br label %8

3:                                                ; preds = %8
  %4 = load { i32 }, ptr %14, align 4
  br label %15

5:                                                ; preds = %8
  %6 = load { i32, i32, ptr }, ptr %14, align 8
  br label %20

7:                                                ; preds = %8
  unreachable

8:                                                ; preds = %2
  %9 = phi { i32, i32, ptr } [ %0, %2 ]
  %10 = phi i32 [ %1, %2 ]
  %11 = call { i16, [16 x i8] } @"core::array::array_at::<core::integer::u32>"({ i32, i32, ptr } %9, i32 %10)
  %12 = extractvalue { i16, [16 x i8] } %11, 0
  %13 = extractvalue { i16, [16 x i8] } %11, 1
  %14 = alloca [16 x i8], i64 1, align 1
  store [16 x i8] %13, ptr %14, align 1
  switch i16 %12, label %7 [
    i16 0, label %3
    i16 1, label %5
  ]

15:                                               ; preds = %3
  %16 = phi { i32 } [ %4, %3 ]
  %17 = call i32 @"struct_deconstruct<Tuple<Box<u32>>>"({ i32 } %16)
  %18 = call { i32 } @"struct_construct<Tuple<u32>>"(i32 %17)
  %19 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"({ i32 } %18)
  ret { i16, [16 x i8] } %19

20:                                               ; preds = %5
  %21 = phi { i32, i32, ptr } [ %6, %5 ]
  %22 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"({ i32, i32, ptr } %21)
  ret { i16, [16 x i8] } %22
}

define void @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(ptr %0, { i32, i32, ptr } %1, i32 %2) {
  %4 = call { i16, [16 x i8] } @"core::array::ArrayIndex::<core::integer::u32>::index"({ i32, i32, ptr } %1, i32 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

define { i16, [16 x i8] } @"core::array::array_at::<core::integer::u32>"({ i32, i32, ptr } %0, i32 %1) {
  br label %7

3:                                                ; preds = %7
  %4 = extractvalue { i32, i32, ptr } %8, 2
  %5 = getelementptr i32, ptr %4, i32 %9
  %6 = load i32, ptr %5, align 4
  br label %12

7:                                                ; preds = %2
  %8 = phi { i32, i32, ptr } [ %0, %2 ]
  %9 = phi i32 [ %1, %2 ]
  %10 = extractvalue { i32, i32, ptr } %8, 0
  %11 = icmp ult i32 %9, %10
  br i1 %11, label %3, label %16

12:                                               ; preds = %3
  %13 = phi i32 [ %6, %3 ]
  %14 = call { i32 } @"struct_construct<Tuple<Box<u32>>>"(i32 %13)
  %15 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"({ i32 } %14)
  ret { i16, [16 x i8] } %15

16:                                               ; preds = %7
  %17 = call { i32, i32, ptr } @"array_new<felt252>"()
  %18 = call { i32, i32, ptr } @"array_append<felt252>"({ i32, i32, ptr } %17, i256 1637570914057682275393755530660268060279989363)
  %19 = call { i16, [16 x i8] } @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"({ i32, i32, ptr } %18)
  ret { i16, [16 x i8] } %19
}

define void @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(ptr %0, { i32, i32, ptr } %1, i32 %2) {
  %4 = call { i16, [16 x i8] } @"core::array::array_at::<core::integer::u32>"({ i32, i32, ptr } %1, i32 %2)
  store { i16, [16 x i8] } %4, ptr %0, align 2
  ret void
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { alwaysinline norecurse nounwind }
attributes #1 = { norecurse nounwind }
attributes #2 = { cold noreturn nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
