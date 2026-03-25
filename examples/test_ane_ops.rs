//! ANE Operation Boundary Tester
//!
//! Uses subprocess isolation per test to survive ANE native crashes (SIGSEGV).
//! Tests both matmul and conv1x1 for training-critical shapes.
//!
//! Run with: cargo run --example test_ane_ops

use std::process::Command;
use std::time::Instant;

fn main() {
    eprintln!();
    eprintln!("  ============================================================");
    eprintln!("  ANE Operation Tester (matmul vs conv1x1)");
    eprintln!("  Subprocess isolation for crash survival");
    eprintln!("  ============================================================");

    // Build the single-test binary first
    eprintln!("\n  Building test binary...");
    let build_status = Command::new("cargo")
        .args(["build", "--example", "test_ane_single"])
        .output()
        .expect("failed to run cargo build");
    if !build_status.status.success() {
        eprintln!(
            "  Build failed: {}",
            String::from_utf8_lossy(&build_status.stderr)
        );
        return;
    }
    eprintln!("  Build OK.");

    // Training-critical shapes:
    // (label, seq_len, in_dim, out_dim)
    let shapes = vec![
        // Small model (dim=64, hidden=128, 2 layers, vocab=512)
        ("sm:qkv", 32, 64, 192),    // QKV fused
        ("sm:attn", 32, 64, 64),    // attention output proj
        ("sm:dual", 32, 64, 256),   // w1+w3 fused (hidden=128)
        ("sm:w2", 32, 256, 64),     // FFN output (hidden=256→dim=64)
        ("sm:logits", 31, 64, 512), // classifier head
        // Medium model (dim=128, hidden=256, 4 layers, vocab=1024)
        ("md:qkv", 32, 128, 384),     // QKV fused
        ("md:attn", 32, 128, 128),    // attention output proj
        ("md:dual", 32, 128, 512),    // w1+w3 fused (hidden=256)
        ("md:w2", 32, 512, 128),      // FFN output
        ("md:logits", 31, 128, 1024), // classifier head
        // Larger seq lens
        ("seq64:qkv", 64, 64, 192),
        ("seq64:dual", 64, 64, 256),
        ("seq64:w2", 64, 256, 64),
        // Edge cases
        ("tiny", 8, 32, 64),
        ("square", 32, 128, 128),
    ];

    eprintln!(
        "\n  {:<12} {:>6} {:>6} {:>6}  | {:>8} {:>8}  | {:>8} {:>8}",
        "shape", "S", "in", "out", "mm_comp", "mm_eval", "cv_comp", "cv_eval"
    );
    eprintln!("  {}", "-".repeat(78));

    for (label, s, i, o) in &shapes {
        let mm = run_test(*s, *i, *o, "matmul");
        let cv = run_test(*s, *i, *o, "conv1x1");

        let mm_s = match &mm {
            TestResult::Ok { compile, eval, .. } => format!("{:>6.0}ms {:>6.1}ms", compile, eval),
            TestResult::Fail(reason) => format!("  {:<14}", reason),
        };
        let cv_s = match &cv {
            TestResult::Ok { compile, eval, .. } => format!("{:>6.0}ms {:>6.1}ms", compile, eval),
            TestResult::Fail(reason) => format!("  {:<14}", reason),
        };

        eprintln!(
            "  {:<12} {:>6} {:>6} {:>6}  | {}  | {}",
            label, s, i, o, mm_s, cv_s
        );
    }

    eprintln!("\n  Done.");
}

enum TestResult {
    Ok {
        compile: f64,
        eval: f64,
        total_ms: f64,
    },
    Fail(String),
}

fn run_test(s: usize, i: usize, o: usize, op: &str) -> TestResult {
    let t0 = Instant::now();

    let output = Command::new("./target/debug/examples/test_ane_single")
        .env("RUSTANE_TEST_S", &s.to_string())
        .env("RUSTANE_TEST_I", &i.to_string())
        .env("RUSTANE_TEST_O", &o.to_string())
        .env("RUSTANE_TEST_OP", op)
        .output();

    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    match output {
        Ok(out) => {
            let status = out.status;
            let stderr = String::from_utf8_lossy(&out.stderr);
            if status.success() {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let parts: Vec<&str> = stdout.trim().split_whitespace().collect();
                if parts.len() >= 3 && parts[0] == "OK" {
                    let cm: f64 = parts[1].parse().unwrap_or(0.0);
                    let em: f64 = parts[2].parse().unwrap_or(0.0);
                    TestResult::Ok {
                        compile: cm,
                        eval: em,
                        total_ms: elapsed,
                    }
                } else {
                    TestResult::Fail("bad stdout".into())
                }
            } else {
                let code = status.code().unwrap_or(-1);
                if stderr.contains("Program Inference error") {
                    TestResult::Fail("inference err".into())
                } else if stderr.contains("compile failed") {
                    TestResult::Fail("compile err".into())
                } else if stderr.contains("all zeros") {
                    TestResult::Fail("all zeros".into())
                } else if code == -11 || code == 139 {
                    TestResult::Fail("SIGSEGV".into())
                } else if code == -6 || code == 134 {
                    TestResult::Fail("SIGABRT".into())
                } else if code == 137 {
                    TestResult::Fail("timeout".into())
                } else {
                    let first_line = stderr.lines().next().unwrap_or("");
                    TestResult::Fail(format!("exit {}", code))
                }
            }
        }
        Err(e) => TestResult::Fail(format!("spawn: {}", e)),
    }
}
