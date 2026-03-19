use crate::common::*;

#[test]
fn mil_graphs_have_expected_layouts() {
    let fused = build_fused_qkv_mil(64, 32);
    assert!(fused.contains("program(1.3)"));
    assert!(fused.contains("tensor<fp32, [1, 64, 1, 32]> x"));
    assert!(fused.contains("tensor<fp16, [1, 64, 1, 32]> x16"));
    assert!(fused.contains("concat(axis = ax, interleave = inter"));
    assert_eq!(fused.matches("conv(").count(), 3);
    assert!(fused.contains("tensor<fp32, [1, 192, 1, 32]> y"));

    let qkt = build_qkt_mil(12, 64, 64);
    assert!(qkt.contains("program(1.3)"));
    assert!(qkt.contains("tensor<fp16, [1, 12, 64, 64]> q"));
    assert!(qkt.contains("tensor<fp16, [1, 12, 64, 64]> k"));
    assert!(qkt.contains("tensor<fp16, [1, 12, 64, 64]> scores"));
    assert!(qkt.contains("matmul(transpose_x = bF, transpose_y = bT"));

    let sv = build_sv_mil(12, 64, 64);
    assert!(sv.contains("program(1.3)"));
    assert!(sv.contains("tensor<fp16, [1, 12, 64, 64]> probs"));
    assert!(sv.contains("tensor<fp16, [1, 12, 64, 64]> v"));
    assert!(sv.contains("tensor<fp16, [1, 12, 64, 64]> out"));
    assert!(sv.contains("matmul(transpose_x = bF, transpose_y = bF"));
}
