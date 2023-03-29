; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define internal { i256, i256 } @"dup<felt252>"(i256 %0) {
  %2 = insertvalue { i256, i256 } undef, i256 %0, 0
  %3 = insertvalue { i256, i256 } %2, i256 %0, 1
  ret { i256, i256 } %3
}

define internal i256 @"store_temp<felt252>"(i256 %0) {
  ret i256 %0
}

define internal i256 @felt252_mul(i256 %0, i256 %1) {
  %3 = zext i256 %0 to i512
  %4 = zext i256 %1 to i512
  %5 = mul i512 %3, %4
  %6 = srem i512 %5, 3618502788666131213697322783095070105623107215331596699973092056135872020481
  %7 = trunc i512 %6 to i256
  br label %8

8:                                                ; preds = %2
  %9 = phi i256 [ %7, %2 ]
  ret i256 %9
}

define internal i256 @"rename<felt252>"(i256 %0) {
  ret i256 %0
}

define i256 @felt_is_zero_felt_is_zero_mul_if_not_zero(i256 %0) {
  br label %2

2:                                                ; preds = %1
  %3 = phi i256 [ %0, %1 ]
  %4 = call { i256, i256 } @"dup<felt252>"(i256 %3)
  %5 = extractvalue { i256, i256 } %4, 0
  %6 = extractvalue { i256, i256 } %4, 1
  %7 = icmp eq i256 %6, 0
  br i1 %7, label %8, label %10

8:                                                ; preds = %2
  %9 = call i256 @"store_temp<felt252>"(i256 0)
  br label %14

10:                                               ; preds = %2
  %11 = phi i256 [ %5, %2 ]
  %12 = call i256 @felt252_mul(i256 %11, i256 2)
  %13 = call i256 @"store_temp<felt252>"(i256 %12)
  br label %14

14:                                               ; preds = %8, %10
  %15 = phi i256 [ %13, %10 ], [ %9, %8 ]
  %16 = call i256 @"rename<felt252>"(i256 %15)
  ret i256 %16
}

define i256 @_mlir_ciface_felt_is_zero_felt_is_zero_mul_if_not_zero(i256 %0) {
  %2 = call i256 @felt_is_zero_felt_is_zero_mul_if_not_zero(i256 %0)
  ret i256 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
