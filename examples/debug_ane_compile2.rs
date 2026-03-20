use rustane::*;

fn main() -> Result<()> {
    // Use a simpler MIL similar to test_ane_linear_minimal that works
    let mil = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremlc-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 128, 1, 64]> x, tensor<fp32, [1, 128, 1, 64]> y) {
        tensor<fp32, [1, 128, 1, 64]> result = add(x=x, y=y)[name = string("add")];
    } -> (result);
}"#;

    println!("Testing simple 2-input MIL");
    let req = ane::ANECompileRequest::new(
        mil,
        vec![128 * 64 * 4, 128 * 64 * 4], // 2 inputs
        vec![128 * 64 * 4],               // 1 output
    );

    match req.compile() {
        Ok(_) => println!("SUCCESS: Simple 2-input MIL compiled"),
        Err(e) => println!("FAILED: {}", e),
    }

    // Now try 3 inputs
    let mil3 = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremlc-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp32, [1, 128, 1, 64]> x, tensor<fp32, [1, 128, 1, 64]> y, tensor<fp32, [1, 128, 1, 1]> z) {
        tensor<fp32, [1, 128, 1, 64]> temp = add(x=x, y=y)[name = string("add")];
        tensor<fp32, [1, 128, 1, 64]> result = mul(x=temp, y=z)[name = string("mul")];
    } -> (result);
}"#;

    println!("\nTesting 3-input MIL");
    let req3 = ane::ANECompileRequest::new(
        mil3,
        vec![128 * 64 * 4, 128 * 64 * 4, 128 * 4], // 3 inputs
        vec![128 * 64 * 4],                        // 1 output
    );

    match req3.compile() {
        Ok(_) => println!("SUCCESS: 3-input MIL compiled"),
        Err(e) => println!("FAILED: {}", e),
    }

    Ok(())
}
