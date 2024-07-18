; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin23.5.0"

@assert_msg_0 = private constant [18 x i8] c"Invalid enum tag.\00"
@assert_msg = private constant [18 x i8] c"Invalid enum tag.\00"

declare void @abort()

declare void @puts(ptr)

define { i64, { i128, [32 x i8] } } @"program::program::main(f1)"(i64 %0) {
  %2 = alloca { i128, [32 x i8] }, i64 1, align 16
  %3 = alloca { i128, [32 x i8] }, i64 1, align 16
  %4 = call { i64, { { i128, i128 }, { i1, [0 x i8] } } } @"core::integer::u256_overflow_mul(f0)"(i64 %0, { i128, i128 } { i128 2, i128 0 }, { i128, i128 } { i128 9, i128 0 })
  %5 = extractvalue { i64, { { i128, i128 }, { i1, [0 x i8] } } } %4, 0
  %6 = extractvalue { i64, { { i128, i128 }, { i1, [0 x i8] } } } %4, 1
  %7 = extractvalue { { i128, i128 }, { i1, [0 x i8] } } %6, 0
  %8 = extractvalue { { i128, i128 }, { i1, [0 x i8] } } %6, 1
  %9 = extractvalue { i1, [0 x i8] } %8, 0
  switch i1 %9, label %10 [
    i1 false, label %12
    i1 true, label %17
  ]

10:                                               ; preds = %1
  br i1 false, label %11, label %21

11:                                               ; preds = %10
  unreachable

12:                                               ; preds = %1
  %13 = insertvalue { i1, { i128, i128 } } { i1 false, { i128, i128 } undef }, { i128, i128 } %7, 1
  store { i1, { i128, i128 } } %13, ptr %3, align 16
  %14 = load { i128, [32 x i8] }, ptr %3, align 16
  %15 = insertvalue { i64, { i128, [32 x i8] } } undef, i64 %5, 0
  %16 = insertvalue { i64, { i128, [32 x i8] } } %15, { i128, [32 x i8] } %14, 1
  ret { i64, { i128, [32 x i8] } } %16

17:                                               ; preds = %1
  store { i1, [0 x i8] } { i1 true, [0 x i8] undef }, ptr %2, align 1
  %18 = load { i128, [32 x i8] }, ptr %2, align 16
  %19 = insertvalue { i64, { i128, [32 x i8] } } undef, i64 %5, 0
  %20 = insertvalue { i64, { i128, [32 x i8] } } %19, { i128, [32 x i8] } %18, 1
  ret { i64, { i128, [32 x i8] } } %20

21:                                               ; preds = %10
  call void @puts(ptr @assert_msg)
  call void @abort()
  unreachable
}

define void @"_mlir_ciface_program::program::main(f1)"(ptr %0, i64 %1) {
  %3 = call { i64, { i128, [32 x i8] } } @"program::program::main(f1)"(i64 %1)
  store { i64, { i128, [32 x i8] } } %3, ptr %0, align 16
  ret void
}

define { i64, { { i128, i128 }, { i1, [0 x i8] } } } @"core::integer::u256_overflow_mul(f0)"(i64 %0, { i128, i128 } %1, { i128, i128 } %2) {
  %4 = extractvalue { i128, i128 } %1, 0
  %5 = extractvalue { i128, i128 } %1, 1
  %6 = extractvalue { i128, i128 } %2, 0
  %7 = extractvalue { i128, i128 } %2, 1
  %8 = zext i128 %4 to i256
  %9 = zext i128 %6 to i256
  %10 = mul i256 %8, %9
  %11 = trunc i256 %10 to i128
  %12 = zext i128 %4 to i256
  %13 = zext i128 %6 to i256
  %14 = mul i256 %12, %13
  %15 = trunc i256 %14 to i128
  %16 = lshr i256 %14, 128
  %17 = trunc i256 %16 to i128
  %18 = zext i128 %4 to i256
  %19 = zext i128 %7 to i256
  %20 = mul i256 %18, %19
  %21 = trunc i256 %20 to i128
  %22 = zext i128 %4 to i256
  %23 = zext i128 %7 to i256
  %24 = mul i256 %22, %23
  %25 = trunc i256 %24 to i128
  %26 = lshr i256 %24, 128
  %27 = trunc i256 %26 to i128
  %28 = zext i128 %5 to i256
  %29 = zext i128 %6 to i256
  %30 = mul i256 %28, %29
  %31 = trunc i256 %30 to i128
  %32 = zext i128 %5 to i256
  %33 = zext i128 %6 to i256
  %34 = mul i256 %32, %33
  %35 = trunc i256 %34 to i128
  %36 = lshr i256 %34, 128
  %37 = trunc i256 %36 to i128
  %38 = add i64 %0, 4
  %39 = call { i128, i1 } @llvm.uadd.with.overflow.i128(i128 %17, i128 %21)
  %40 = extractvalue { i128, i1 } %39, 0
  %41 = extractvalue { i128, i1 } %39, 1
  br i1 %41, label %55, label %42

42:                                               ; preds = %3
  %43 = phi i64 [ %38, %3 ]
  %44 = icmp eq i128 %27, 0
  br i1 %44, label %45, label %67

45:                                               ; preds = %42
  %46 = phi i128 [ %37, %42 ]
  %47 = phi i128 [ 0, %42 ]
  %48 = icmp ne i128 %46, %47
  switch i1 %48, label %49 [
    i1 false, label %51
    i1 true, label %67
  ]

49:                                               ; preds = %45
  br i1 false, label %50, label %94

50:                                               ; preds = %49
  unreachable

51:                                               ; preds = %45
  %52 = add i64 %43, 1
  %53 = call { i128, i1 } @llvm.usub.with.overflow.i128(i128 0, i128 %5)
  %54 = extractvalue { i128, i1 } %53, 1
  br i1 %54, label %59, label %55

55:                                               ; preds = %51, %3
  %56 = phi i1 [ false, %51 ], [ true, %3 ]
  %57 = phi i64 [ %52, %51 ], [ %38, %3 ]
  %58 = insertvalue { i1, [0 x i8] } undef, i1 %56, 0
  br label %72

59:                                               ; preds = %51
  %60 = phi i64 [ %52, %51 ]
  %61 = phi i128 [ 0, %51 ]
  %62 = phi i128 [ %7, %51 ]
  %63 = add i64 %60, 1
  %64 = call { i128, i1 } @llvm.usub.with.overflow.i128(i128 %61, i128 %62)
  %65 = extractvalue { i128, i1 } %64, 1
  %66 = insertvalue { i1, [0 x i8] } undef, i1 %65, 0
  br label %72

67:                                               ; preds = %45, %42
  %68 = phi i128 [ %11, %45 ], [ %11, %42 ]
  %69 = phi i128 [ %31, %45 ], [ %31, %42 ]
  %70 = phi i64 [ %43, %45 ], [ %43, %42 ]
  %71 = phi i128 [ %40, %45 ], [ %40, %42 ]
  br label %72

72:                                               ; preds = %55, %59, %67
  %73 = phi i128 [ %68, %67 ], [ %11, %59 ], [ %11, %55 ]
  %74 = phi i128 [ %69, %67 ], [ %31, %59 ], [ %31, %55 ]
  %75 = phi i64 [ %70, %67 ], [ %63, %59 ], [ %57, %55 ]
  %76 = phi i128 [ %71, %67 ], [ %40, %59 ], [ %40, %55 ]
  %77 = phi { i1, [0 x i8] } [ { i1 true, [0 x i8] undef }, %67 ], [ %66, %59 ], [ %58, %55 ]
  %78 = add i64 %75, 1
  %79 = call { i128, i1 } @llvm.uadd.with.overflow.i128(i128 %76, i128 %74)
  %80 = extractvalue { i128, i1 } %79, 0
  %81 = extractvalue { i128, i1 } %79, 1
  br i1 %81, label %82, label %83

82:                                               ; preds = %72
  br label %83

83:                                               ; preds = %82, %72
  %84 = phi i128 [ %73, %82 ], [ %73, %72 ]
  %85 = phi i64 [ %78, %82 ], [ %78, %72 ]
  %86 = phi i128 [ %80, %82 ], [ %80, %72 ]
  %87 = phi { i1, [0 x i8] } [ { i1 true, [0 x i8] undef }, %82 ], [ %77, %72 ]
  %88 = insertvalue { i128, i128 } undef, i128 %84, 0
  %89 = insertvalue { i128, i128 } %88, i128 %86, 1
  %90 = insertvalue { { i128, i128 }, { i1, [0 x i8] } } undef, { i128, i128 } %89, 0
  %91 = insertvalue { { i128, i128 }, { i1, [0 x i8] } } %90, { i1, [0 x i8] } %87, 1
  %92 = insertvalue { i64, { { i128, i128 }, { i1, [0 x i8] } } } undef, i64 %85, 0
  %93 = insertvalue { i64, { { i128, i128 }, { i1, [0 x i8] } } } %92, { { i128, i128 }, { i1, [0 x i8] } } %91, 1
  ret { i64, { { i128, i128 }, { i1, [0 x i8] } } } %93

94:                                               ; preds = %49
  call void @puts(ptr @assert_msg_0)
  call void @abort()
  unreachable
}

define void @"_mlir_ciface_core::integer::u256_overflow_mul(f0)"(ptr %0, i64 %1, { i128, i128 } %2, { i128, i128 } %3) {
  %5 = call { i64, { { i128, i128 }, { i1, [0 x i8] } } } @"core::integer::u256_overflow_mul(f0)"(i64 %1, { i128, i128 } %2, { i128, i128 } %3)
  store { i64, { { i128, i128 }, { i1, [0 x i8] } } } %5, ptr %0, align 16
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i128, i1 } @llvm.uadd.with.overflow.i128(i128, i128) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i128, i1 } @llvm.usub.with.overflow.i128(i128, i128) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
