; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@assert_msg_2 = private constant [18 x i8] c"Invalid enum tag.\00"
@assert_msg_1 = private constant [18 x i8] c"Invalid enum tag.\00"
@assert_msg_0 = private constant [18 x i8] c"Invalid enum tag.\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @abort()

declare void @puts(ptr)

declare ptr @realloc(ptr, i64)

define { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } @"hello_starknet::hello_starknet::Echo::__wrapper__echo"([0 x i8] %0, i128 %1, ptr %2, { { ptr, i32, i32 } } %3) {
  %5 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %6 = alloca { i1, [7 x i8], i252, [0 x i8] }, i64 1, align 8
  %7 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %8 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %9 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %10 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %11 = icmp uge i128 %1, 4070
  %12 = call i128 @llvm.usub.sat.i128(i128 %1, i128 4070)
  br i1 %11, label %13, label %27

13:                                               ; preds = %4
  %14 = phi { { ptr, i32, i32 } } [ %3, %4 ]
  %15 = call { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } @"core::Felt252Serde::deserialize"({ { ptr, i32, i32 } } %14)
  %16 = extractvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %15, 0
  %17 = extractvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %15, 1
  store { i1, [7 x i8], i252, [0 x i8] } %17, ptr %6, align 8
  %18 = extractvalue { i1, [7 x i8], i252, [0 x i8] } %17, 0
  switch i1 %18, label %19 [
    i1 false, label %21
    i1 true, label %27
  ]

19:                                               ; preds = %13
  br i1 false, label %20, label %70

20:                                               ; preds = %19
  unreachable

21:                                               ; preds = %13
  %22 = load { i1, [7 x i8], i252 }, ptr %6, align 4
  %23 = extractvalue { i1, [7 x i8], i252 } %22, 2
  %24 = extractvalue { { ptr, i32, i32 } } %16, 0
  %25 = extractvalue { ptr, i32, i32 } %24, 1
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %44, label %27

27:                                               ; preds = %44, %21, %13, %4
  %28 = phi i252 [ 375233589013918064796019, %44 ], [ 7733229381460288120802334208475838166080759535023995805565484692595, %21 ], [ 485748461484230571791265682659113160264223489397539653310998840191492913, %13 ], [ 375233589013918064796019, %4 ]
  %29 = phi ptr [ %8, %44 ], [ %10, %21 ], [ %7, %13 ], [ %5, %4 ]
  %30 = phi ptr [ %8, %44 ], [ %10, %21 ], [ %7, %13 ], [ %5, %4 ]
  %31 = phi i128 [ %46, %44 ], [ %12, %21 ], [ %12, %13 ], [ %12, %4 ]
  %32 = call ptr @realloc(ptr null, i64 256)
  %33 = insertvalue { ptr, i32, i32 } zeroinitializer, ptr %32, 0
  %34 = insertvalue { ptr, i32, i32 } %33, i32 8, 2
  %35 = getelementptr i252, ptr %32, i32 0
  store i252 %28, ptr %35, align 4
  %36 = insertvalue { ptr, i32, i32 } %34, i32 1, 1
  %37 = insertvalue { {}, { ptr, i32, i32 } } undef, { ptr, i32, i32 } %36, 1
  %38 = insertvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } } } { i1 true, [7 x i8] undef, { {}, { ptr, i32, i32 } } undef }, { {}, { ptr, i32, i32 } } %37, 2
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } } } %38, ptr %29, align 8
  %39 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %30, align 8
  %40 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } undef, [0 x i8] %0, 0
  %41 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %40, i128 %31, 1
  %42 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %41, ptr %2, 2
  %43 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %42, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %39, 3
  ret { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %43

44:                                               ; preds = %21
  %45 = icmp uge i128 %12, 0
  %46 = call i128 @llvm.usub.sat.i128(i128 %12, i128 0)
  br i1 %45, label %47, label %27

47:                                               ; preds = %44
  %48 = call { { {} }, i252 } @"hello_starknet::hello_starknet::Echo::echo"({ {} } undef, i252 %23)
  %49 = extractvalue { { {} }, i252 } %48, 0
  %50 = extractvalue { { {} }, i252 } %48, 1
  %51 = call { { ptr, i32, i32 }, {} } @"core::Felt252Serde::serialize"(i252 %50, { ptr, i32, i32 } zeroinitializer)
  %52 = extractvalue { { ptr, i32, i32 }, {} } %51, 0
  %53 = extractvalue { { ptr, i32, i32 }, {} } %51, 1
  %54 = extractvalue { ptr, i32, i32 } %52, 0
  %55 = extractvalue { ptr, i32, i32 } %52, 1
  %56 = zext i32 %55 to i64
  %57 = mul i64 %56, 32
  %58 = call ptr @realloc(ptr null, i64 %57)
  call void @llvm.memcpy.p0.p0.i64(ptr %58, ptr %54, i64 %57, i1 false)
  %59 = insertvalue { ptr, i32, i32 } undef, ptr %58, 0
  %60 = insertvalue { ptr, i32, i32 } %59, i32 %55, 1
  %61 = insertvalue { ptr, i32, i32 } %60, i32 %55, 2
  %62 = insertvalue { { ptr, i32, i32 } } undef, { ptr, i32, i32 } %61, 0
  %63 = insertvalue { { { ptr, i32, i32 } } } undef, { { ptr, i32, i32 } } %62, 0
  %64 = insertvalue { i1, [7 x i8], { { { ptr, i32, i32 } } } } { i1 false, [7 x i8] undef, { { { ptr, i32, i32 } } } undef }, { { { ptr, i32, i32 } } } %63, 2
  store { i1, [7 x i8], { { { ptr, i32, i32 } } } } %64, ptr %9, align 8
  %65 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %9, align 8
  %66 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } undef, [0 x i8] %0, 0
  %67 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %66, i128 %46, 1
  %68 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %67, ptr %2, 2
  %69 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %68, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %65, 3
  ret { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %69

70:                                               ; preds = %19
  call void @puts(ptr @assert_msg_0)
  call void @abort()
  unreachable
}

define void @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__echo"(ptr %0, [0 x i8] %1, i128 %2, ptr %3, { { ptr, i32, i32 } } %4) {
  %6 = call { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } @"hello_starknet::hello_starknet::Echo::__wrapper__echo"([0 x i8] %1, i128 %2, ptr %3, { { ptr, i32, i32 } } %4)
  store { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %6, ptr %0, align 8
  ret void
}

define { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } @"hello_starknet::hello_starknet::Echo::__wrapper__constructor"([0 x i8] %0, i128 %1, ptr %2, { { ptr, i32, i32 } } %3) {
  %5 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %6 = alloca { i1, [7 x i8], i252, [0 x i8] }, i64 1, align 8
  %7 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %8 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %9 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %10 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %11 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %12 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %13 = icmp uge i128 %1, 3970
  %14 = call i128 @llvm.usub.sat.i128(i128 %1, i128 3970)
  br i1 %13, label %15, label %29

15:                                               ; preds = %4
  %16 = phi { { ptr, i32, i32 } } [ %3, %4 ]
  %17 = call { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } @"core::Felt252Serde::deserialize"({ { ptr, i32, i32 } } %16)
  %18 = extractvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %17, 0
  %19 = extractvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %17, 1
  store { i1, [7 x i8], i252, [0 x i8] } %19, ptr %6, align 8
  %20 = extractvalue { i1, [7 x i8], i252, [0 x i8] } %19, 0
  switch i1 %20, label %21 [
    i1 false, label %23
    i1 true, label %29
  ]

21:                                               ; preds = %49, %15
  br i1 false, label %22, label %74

22:                                               ; preds = %21
  unreachable

23:                                               ; preds = %15
  %24 = load { i1, [7 x i8], i252 }, ptr %6, align 4
  %25 = extractvalue { i1, [7 x i8], i252 } %24, 2
  %26 = extractvalue { { ptr, i32, i32 } } %18, 0
  %27 = extractvalue { ptr, i32, i32 } %26, 1
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %46, label %29

29:                                               ; preds = %46, %23, %15, %4
  %30 = phi i252 [ 375233589013918064796019, %46 ], [ 7733229381460288120802334208475838166080759535023995805565484692595, %23 ], [ 485748461484230571791265682659113160264223489397539653310998840191492913, %15 ], [ 375233589013918064796019, %4 ]
  %31 = phi ptr [ %8, %46 ], [ %12, %23 ], [ %7, %15 ], [ %5, %4 ]
  %32 = phi ptr [ %8, %46 ], [ %12, %23 ], [ %7, %15 ], [ %5, %4 ]
  %33 = phi i128 [ %48, %46 ], [ %14, %23 ], [ %14, %15 ], [ %14, %4 ]
  %34 = call ptr @realloc(ptr null, i64 256)
  %35 = insertvalue { ptr, i32, i32 } zeroinitializer, ptr %34, 0
  %36 = insertvalue { ptr, i32, i32 } %35, i32 8, 2
  %37 = getelementptr i252, ptr %34, i32 0
  store i252 %30, ptr %37, align 4
  %38 = insertvalue { ptr, i32, i32 } %36, i32 1, 1
  %39 = insertvalue { {}, { ptr, i32, i32 } } undef, { ptr, i32, i32 } %38, 1
  %40 = insertvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } } } { i1 true, [7 x i8] undef, { {}, { ptr, i32, i32 } } undef }, { {}, { ptr, i32, i32 } } %39, 2
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } } } %40, ptr %31, align 8
  %41 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %32, align 8
  %42 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } undef, [0 x i8] %0, 0
  %43 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %42, i128 %33, 1
  %44 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %43, ptr %2, 2
  %45 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %44, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %41, 3
  ret { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %45

46:                                               ; preds = %23
  %47 = icmp uge i128 %14, 0
  %48 = call i128 @llvm.usub.sat.i128(i128 %14, i128 0)
  br i1 %47, label %49, label %29

49:                                               ; preds = %46
  %50 = call { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } @"hello_starknet::hello_starknet::Echo::constructor"({ {} } undef, i252 %25)
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %50, ptr %9, align 8
  %51 = extractvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %50, 0
  switch i1 %51, label %21 [
    i1 false, label %52
    i1 true, label %65
  ]

52:                                               ; preds = %49
  %53 = call ptr @realloc(ptr null, i64 0)
  call void @llvm.memcpy.p0.p0.i64(ptr %53, ptr null, i64 0, i1 false)
  %54 = insertvalue { ptr, i32, i32 } undef, ptr %53, 0
  %55 = insertvalue { ptr, i32, i32 } %54, i32 0, 1
  %56 = insertvalue { ptr, i32, i32 } %55, i32 0, 2
  %57 = insertvalue { { ptr, i32, i32 } } undef, { ptr, i32, i32 } %56, 0
  %58 = insertvalue { { { ptr, i32, i32 } } } undef, { { ptr, i32, i32 } } %57, 0
  %59 = insertvalue { i1, [7 x i8], { { { ptr, i32, i32 } } } } { i1 false, [7 x i8] undef, { { { ptr, i32, i32 } } } undef }, { { { ptr, i32, i32 } } } %58, 2
  store { i1, [7 x i8], { { { ptr, i32, i32 } } } } %59, ptr %11, align 8
  %60 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %11, align 8
  %61 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } undef, [0 x i8] %0, 0
  %62 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %61, i128 %48, 1
  %63 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %62, ptr %2, 2
  %64 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %63, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %60, 3
  ret { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %64

65:                                               ; preds = %49
  %66 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } } }, ptr %9, align 8
  %67 = extractvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } } } %66, 2
  %68 = insertvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } } } { i1 true, [7 x i8] undef, { {}, { ptr, i32, i32 } } undef }, { {}, { ptr, i32, i32 } } %67, 2
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } } } %68, ptr %10, align 8
  %69 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %10, align 8
  %70 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } undef, [0 x i8] %0, 0
  %71 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %70, i128 %48, 1
  %72 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %71, ptr %2, 2
  %73 = insertvalue { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %72, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %69, 3
  ret { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %73

74:                                               ; preds = %21
  call void @puts(ptr @assert_msg_1)
  call void @abort()
  unreachable
}

define void @"_mlir_ciface_hello_starknet::hello_starknet::Echo::__wrapper__constructor"(ptr %0, [0 x i8] %1, i128 %2, ptr %3, { { ptr, i32, i32 } } %4) {
  %6 = call { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } @"hello_starknet::hello_starknet::Echo::__wrapper__constructor"([0 x i8] %1, i128 %2, ptr %3, { { ptr, i32, i32 } } %4)
  store { [0 x i8], i128, ptr, { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } } %6, ptr %0, align 8
  ret void
}

define { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } @"core::Felt252Serde::deserialize"({ { ptr, i32, i32 } } %0) {
  %2 = alloca { i1, [7 x i8], ptr, [0 x i8] }, i64 1, align 8
  %3 = alloca { i1, [7 x i8], ptr, [0 x i8] }, i64 1, align 8
  %4 = alloca { i1, [7 x i8], i252, [0 x i8] }, i64 1, align 8
  %5 = alloca { i1, [7 x i8], i252, [0 x i8] }, i64 1, align 8
  %6 = alloca { i1, [7 x i8], ptr, [0 x i8] }, i64 1, align 8
  %7 = extractvalue { { ptr, i32, i32 } } %0, 0
  %8 = extractvalue { ptr, i32, i32 } %7, 1
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %22, label %10

10:                                               ; preds = %1
  %11 = extractvalue { ptr, i32, i32 } %7, 0
  %12 = getelementptr i252, ptr %11, i32 0
  %13 = call ptr @realloc(ptr null, i64 32)
  %14 = load i252, ptr %12, align 8
  store i252 %14, ptr %13, align 8
  %15 = getelementptr i252, ptr %11, i32 1
  %16 = sub i32 %8, 1
  %17 = zext i32 %16 to i64
  %18 = mul i64 %17, 32
  call void @llvm.memmove.p0.p0.i64(ptr %11, ptr %15, i64 %18, i1 false)
  %19 = insertvalue { ptr, i32, i32 } %7, i32 %16, 1
  %20 = insertvalue { i1, [7 x i8], ptr } { i1 false, [7 x i8] undef, ptr undef }, ptr %13, 2
  store { i1, [7 x i8], ptr } %20, ptr %6, align 8
  %21 = load { i1, [7 x i8], ptr, [0 x i8] }, ptr %6, align 8
  br label %24

22:                                               ; preds = %1
  store { i1, [0 x i8], {} } { i1 true, [0 x i8] undef, {} undef }, ptr %2, align 8
  %23 = load { i1, [7 x i8], ptr, [0 x i8] }, ptr %2, align 8
  br label %24

24:                                               ; preds = %22, %10
  %25 = phi { ptr, i32, i32 } [ %7, %22 ], [ %19, %10 ]
  %26 = phi { i1, [7 x i8], ptr, [0 x i8] } [ %23, %22 ], [ %21, %10 ]
  %27 = insertvalue { { ptr, i32, i32 } } undef, { ptr, i32, i32 } %25, 0
  store { i1, [7 x i8], ptr, [0 x i8] } %26, ptr %3, align 8
  %28 = extractvalue { i1, [7 x i8], ptr, [0 x i8] } %26, 0
  switch i1 %28, label %29 [
    i1 false, label %31
    i1 true, label %39
  ]

29:                                               ; preds = %24
  br i1 false, label %30, label %43

30:                                               ; preds = %29
  unreachable

31:                                               ; preds = %24
  %32 = load { i1, [7 x i8], ptr }, ptr %3, align 8
  %33 = extractvalue { i1, [7 x i8], ptr } %32, 2
  %34 = load i252, ptr %33, align 8
  %35 = insertvalue { i1, [7 x i8], i252 } { i1 false, [7 x i8] undef, i252 undef }, i252 %34, 2
  store { i1, [7 x i8], i252 } %35, ptr %5, align 8
  %36 = load { i1, [7 x i8], i252, [0 x i8] }, ptr %5, align 8
  %37 = insertvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } undef, { { ptr, i32, i32 } } %27, 0
  %38 = insertvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %37, { i1, [7 x i8], i252, [0 x i8] } %36, 1
  ret { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %38

39:                                               ; preds = %24
  store { i1, [0 x i8], {} } { i1 true, [0 x i8] undef, {} undef }, ptr %4, align 8
  %40 = load { i1, [7 x i8], i252, [0 x i8] }, ptr %4, align 8
  %41 = insertvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } undef, { { ptr, i32, i32 } } %27, 0
  %42 = insertvalue { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %41, { i1, [7 x i8], i252, [0 x i8] } %40, 1
  ret { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %42

43:                                               ; preds = %29
  call void @puts(ptr @assert_msg_2)
  call void @abort()
  unreachable
}

define void @"_mlir_ciface_core::Felt252Serde::deserialize"(ptr %0, { { ptr, i32, i32 } } %1) {
  %3 = call { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } @"core::Felt252Serde::deserialize"({ { ptr, i32, i32 } } %1)
  store { { { ptr, i32, i32 } }, { i1, [7 x i8], i252, [0 x i8] } } %3, ptr %0, align 8
  ret void
}

define { { {} }, i252 } @"hello_starknet::hello_starknet::Echo::echo"({ {} } %0, i252 %1) {
  %3 = insertvalue { { {} }, i252 } undef, { {} } %0, 0
  %4 = insertvalue { { {} }, i252 } %3, i252 %1, 1
  ret { { {} }, i252 } %4
}

define void @"_mlir_ciface_hello_starknet::hello_starknet::Echo::echo"(ptr %0, { {} } %1, i252 %2) {
  %4 = call { { {} }, i252 } @"hello_starknet::hello_starknet::Echo::echo"({ {} } %1, i252 %2)
  store { { {} }, i252 } %4, ptr %0, align 4
  ret void
}

define { { ptr, i32, i32 }, {} } @"core::Felt252Serde::serialize"(i252 %0, { ptr, i32, i32 } %1) {
  %3 = extractvalue { ptr, i32, i32 } %1, 0
  %4 = extractvalue { ptr, i32, i32 } %1, 1
  %5 = extractvalue { ptr, i32, i32 } %1, 2
  %6 = icmp uge i32 %4, %5
  br i1 %6, label %7, label %15

7:                                                ; preds = %2
  %8 = add i32 %5, %5
  %9 = call i32 @llvm.umax.i32(i32 %8, i32 8)
  %10 = zext i32 %9 to i64
  %11 = mul i64 %10, 32
  %12 = call ptr @realloc(ptr %3, i64 %11)
  %13 = insertvalue { ptr, i32, i32 } %1, ptr %12, 0
  %14 = insertvalue { ptr, i32, i32 } %13, i32 %9, 2
  br label %16

15:                                               ; preds = %2
  br label %16

16:                                               ; preds = %7, %15
  %17 = phi { ptr, i32, i32 } [ %1, %15 ], [ %14, %7 ]
  %18 = phi ptr [ %3, %15 ], [ %12, %7 ]
  br label %19

19:                                               ; preds = %16
  %20 = getelementptr i252, ptr %18, i32 %4
  store i252 %0, ptr %20, align 4
  %21 = add i32 %4, 1
  %22 = insertvalue { ptr, i32, i32 } %17, i32 %21, 1
  %23 = insertvalue { { ptr, i32, i32 }, {} } undef, { ptr, i32, i32 } %22, 0
  %24 = insertvalue { { ptr, i32, i32 }, {} } %23, {} undef, 1
  ret { { ptr, i32, i32 }, {} } %24
}

define void @"_mlir_ciface_core::Felt252Serde::serialize"(ptr %0, i252 %1, { ptr, i32, i32 } %2) {
  %4 = call { { ptr, i32, i32 }, {} } @"core::Felt252Serde::serialize"(i252 %1, { ptr, i32, i32 } %2)
  store { { ptr, i32, i32 }, {} } %4, ptr %0, align 8
  ret void
}

define { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } @"hello_starknet::hello_starknet::Echo::constructor"({ {} } %0, i252 %1) {
  %3 = alloca { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, i64 1, align 8
  %4 = call ptr @realloc(ptr null, i64 256)
  %5 = insertvalue { ptr, i32, i32 } zeroinitializer, ptr %4, 0
  %6 = insertvalue { ptr, i32, i32 } %5, i32 8, 2
  %7 = getelementptr i252, ptr %4, i32 0
  store i252 482670963043, ptr %7, align 4
  %8 = insertvalue { ptr, i32, i32 } %6, i32 1, 1
  %9 = insertvalue { {}, { ptr, i32, i32 } } undef, { ptr, i32, i32 } %8, 1
  %10 = insertvalue { i1, [7 x i8], { {}, { ptr, i32, i32 } } } { i1 true, [7 x i8] undef, { {}, { ptr, i32, i32 } } undef }, { {}, { ptr, i32, i32 } } %9, 2
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } } } %10, ptr %3, align 8
  %11 = load { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] }, ptr %3, align 8
  ret { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %11
}

define void @"_mlir_ciface_hello_starknet::hello_starknet::Echo::constructor"(ptr %0, { {} } %1, i252 %2) {
  %4 = call { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } @"hello_starknet::hello_starknet::Echo::constructor"({ {} } %1, i252 %2)
  store { i1, [7 x i8], { {}, { ptr, i32, i32 } }, [0 x i8] } %4, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i128 @llvm.usub.sat.i128(i128, i128) #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
