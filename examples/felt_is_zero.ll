; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @dprintf(i32, ptr, ...)

define internal i256 @felt252_mul(i256 %0, i256 %1) {
  %3 = zext i256 %0 to i512
  %4 = zext i256 %1 to i512
  %5 = mul i512 %3, %4
  %6 = urem i512 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %7 = trunc i512 %6 to i256
  ret i256 %7
}

define i256 @"felt_is_zero::felt_is_zero::mul_if_not_zero"(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = icmp eq i256 %3, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %2
  br label %9

6:                                                ; preds = %2
  %7 = phi i256 [ %3, %2 ]
  %8 = call i256 @felt252_mul(i256 %7, i256 2)
  br label %9

9:                                                ; preds = %5, %6
  %10 = phi i256 [ %8, %6 ], [ 0, %5 ]
  ret i256 %10
}

define i256 @"_mlir_ciface_felt_is_zero::felt_is_zero::mul_if_not_zero"(i256 %0) {
  %2 = call i256 @"felt_is_zero::felt_is_zero::mul_if_not_zero"(i256 %0)
  ret i256 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
