//! End-to-end hybrid training step: ANE forward + ANE backward + CPU weight update
//!
//! Measures complete training step time for a multi-layer SwiGLU FFN,
//! comparing ANE-assisted hybrid vs pure CPU BLAS.
//!
//! ANE handles: forward matmuls (conv1x1), backward input grad matmuls (conv1x1)
//! CPU handles: SiLU/sigmoid element-wise, weight gradients (BLAS), SGD update
//!
//! Usage: cargo run --release --example train_hybrid_step -- [D] [SP] [layers] [steps]

use half::f16;
use std::env;
use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
}
const ROW: i32 = 101;
const NT: i32 = 111;
const TR: i32 = 112;

fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            NT,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}
fn mm_at(a: &[f32], k: usize, m: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            TR,
            NT,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}
fn mm_abt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            ROW,
            NT,
            TR,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

fn build_blob(w: &[f32]) -> Vec<u8> {
    let ws = w.len() * 2;
    let t = 128 + ws;
    let mut b = vec![0u8; t];
    b[0] = 1;
    b[4] = 2;
    b[64] = 0xEF;
    b[65] = 0xBE;
    b[66] = 0xAD;
    b[67] = 0xDE;
    b[68] = 1;
    b[72..76].copy_from_slice(&(ws as u32).to_le_bytes());
    b[80..84].copy_from_slice(&128u32.to_le_bytes());
    for (i, &v) in w.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[128 + i * 2] = (h & 0xFF) as u8;
        b[128 + i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn to_fp16(d: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; d.len() * 2];
    for (i, &v) in d.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        b[i * 2] = (h & 0xFF) as u8;
        b[i * 2 + 1] = (h >> 8) as u8;
    }
    b
}
fn from_fp16(r: &[u8]) -> Vec<f32> {
    let mut o = vec![0.0f32; r.len() / 2];
    for i in 0..o.len() {
        let h = (r[i * 2] as u16) | ((r[i * 2 + 1] as u16) << 8);
        o[i] = f16::from_bits(h).to_f32();
    }
    o
}
fn rand_m(r: usize, c: usize, s: f32, seed: u64) -> Vec<f32> {
    (0..r * c)
        .map(|i| {
            let x = ((i as u64 * 2654435761)
                .wrapping_add(seed)
                .wrapping_mul(0x9E3779B97F4A7C15)
                >> 33) as f32
                / (1u64 << 31) as f32;
            (x - 0.5) * 2.0 * s
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════
// MIL GENERATION
// ═══════════════════════════════════════════════════════════════

fn mil_header() -> &'static str {
    "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
}

/// Forward: x → gate, up, y (3 outputs via residual concat trick)
fn mil_forward(d: usize, inter: usize, sp: usize) -> String {
    let total_ic = inter + d;
    let mut m = String::new();
    m.push_str(mil_header());
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> x) {\n");
    // 3 weights
    for (wn, oc, ic) in [("Wg", inter, d), ("Wu", inter, d), ("WdI", d, total_ic)] {
        m.push_str("        tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]> ");
        m.push_str(wn);
        m.push_str(" = const()[name = tensor<string, []>(\"");
        m.push_str(wn);
        m.push_str("\"), val = tensor<fp16, [");
        m.push_str(&oc.to_string());
        m.push_str(", ");
        m.push_str(&ic.to_string());
        m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/");
        m.push_str(wn);
        m.push_str(".bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    }
    // conv params
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    let cp = |m: &mut String, s: &str| m.push_str(s);
    // gate = Wg @ x
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &inter.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> gate = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wg, x = x);\n");
    // up = Wu @ x
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &inter.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> up = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = Wu, x = x);\n");
    // silu = gate * sigmoid(gate)
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &inter.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> sig = sigmoid(x = gate);\n");
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &inter.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> fused = mul(x = gate, y = sig);\n");
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &inter.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> act = mul(x = fused, y = up);\n");
    // concat + down with residual
    cp(&mut m, "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    cp(&mut m, "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &total_ic.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(
        &mut m,
        "]> cat = concat(values = (act, x), axis = ax, interleave = ci);\n",
    );
    cp(&mut m, "        tensor<fp16, [1, ");
    cp(&mut m, &d.to_string());
    cp(&mut m, ", 1, ");
    cp(&mut m, &sp.to_string());
    cp(&mut m, "]> y = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdI, x = cat);\n");
    m.push_str("    } -> (gate, up, y);\n}\n");
    m
}

/// Backward Program A: dfused = Wd^T @ dy
fn mil_bwd_dfused(d: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(mil_header());
    m.push_str("    func main<ios16>(tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dy) {\n");
    m.push_str("        tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]> WdT = const()[name = tensor<string, []>(\"WdT\"), val = tensor<fp16, [");
    m.push_str(&inter.to_string());
    m.push_str(", ");
    m.push_str(&d.to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/WdT.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dfused = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = WdT, x = dy);\n");
    m.push_str("    } -> (dfused);\n}\n");
    m
}

/// Backward Program B: dx_partial = [WgT|WuT] @ concat(dgate, dup)
fn mil_bwd_dx(d: usize, inter: usize, sp: usize) -> String {
    let mut m = String::new();
    m.push_str(mil_header());
    m.push_str("    func main<ios16>(\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dgate,\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&inter.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dup\n");
    m.push_str("    ) {\n");
    m.push_str("        tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, 1]> RW = const()[name = tensor<string, []>(\"RW\"), val = tensor<fp16, [");
    m.push_str(&d.to_string());
    m.push_str(", ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/RW.bin\"), offset = tensor<uint64, []>(64)))]  ;\n");
    m.push_str("        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n");
    m.push_str("        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n");
    m.push_str("        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n");
    m.push_str("        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n");
    m.push_str("        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(1)];\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&(2 * inter).to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> cat = concat(values = (dgate, dup), axis = ax, interleave = ci);\n");
    m.push_str("        tensor<fp16, [1, ");
    m.push_str(&d.to_string());
    m.push_str(", 1, ");
    m.push_str(&sp.to_string());
    m.push_str("]> dx = conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, weight = RW, x = cat);\n");
    m.push_str("    } -> (dx);\n}\n");
    m
}

// ═══════════════════════════════════════════════════════════════
// PER-LAYER STATE
// ═══════════════════════════════════════════════════════════════

struct FfnLayer {
    wg: Vec<f32>,
    wu: Vec<f32>,
    wd: Vec<f32>,
    d: usize,
    inter: usize,
    sp: usize,
    // ANE executors
    fwd: rustane::wrapper::ANEExecutor,
    bwd_a: rustane::wrapper::ANEExecutor,
    bwd_b: rustane::wrapper::ANEExecutor,
    // Cached fp16 inputs
    dy16: Vec<u8>,
    // Cached backward intermediates (CPU)
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl FfnLayer {
    fn new(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        let wg = rand_m(inter, d, 0.02, seed);
        let wu = rand_m(inter, d, 0.02, seed + 1);
        let wd = rand_m(d, inter, 0.02, seed + 2);

        // Forward weight: WdI = [Wd | I] concatenated along input channels
        let mut wdi = vec![0.0f32; d * (inter + d)];
        for r in 0..d {
            for c in 0..inter {
                wdi[r * (inter + d) + c] = wd[r * inter + c];
            }
            wdi[r * (inter + d) + inter + r] = 1.0; // identity at position inter+r
        }

        // Backward weights
        // WdT: [inter, d] for conv1x1(out=inter, in=d)
        let mut wdt = vec![0.0f32; inter * d];
        for i in 0..inter {
            for j in 0..d {
                wdt[i * d + j] = wd[j * inter + i];
            }
        }
        // RW = [WgT | WuT]: [d, 2*inter]
        let mut wgt = vec![0.0f32; d * inter];
        for r in 0..d {
            for c in 0..inter {
                wgt[r * inter + c] = wg[c * d + r];
            }
        }
        let mut wut = vec![0.0f32; d * inter];
        for r in 0..d {
            for c in 0..inter {
                wut[r * inter + c] = wu[c * d + r];
            }
        }
        let mut rw = vec![0.0f32; d * 2 * inter];
        for r in 0..d {
            for c in 0..inter {
                rw[r * (2 * inter) + c] = wgt[r * inter + c];
            }
            for c in 0..inter {
                rw[r * (2 * inter) + inter + c] = wut[r * inter + c];
            }
        }

        // Build blobs
        let blob_wg = build_blob(&wg);
        let blob_wu = build_blob(&wu);
        let blob_wdi = build_blob(&wdi);
        let blob_wdt = build_blob(&wdt);
        let blob_rw = build_blob(&rw);

        // Compile forward
        let mil_fwd = mil_forward(d, inter, sp);
        let mut fwd = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_fwd,
                &[
                    "@model_path/weights/Wg.bin",
                    "@model_path/weights/Wu.bin",
                    "@model_path/weights/WdI.bin",
                ],
                &[&blob_wg[..], &blob_wu[..], &blob_wdi[..]],
                &[blob_wg.len(), blob_wu.len(), blob_wdi.len()],
                &[d * sp * 2],
                &[inter * sp * 2, inter * sp * 2, d * sp * 2],
            )
            .expect("fwd compile");

        // Compile backward A: Wd^T @ dy
        let mil_a = mil_bwd_dfused(d, inter, sp);
        let mut bwd_a = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_a,
                &["@model_path/weights/WdT.bin"],
                &[&blob_wdt[..]],
                &[blob_wdt.len()],
                &[d * sp * 2],
                &[inter * sp * 2],
            )
            .expect("bwd_a compile");

        // Compile backward B: [WgT|WuT] @ concat(dgate, dup)
        let mil_b = mil_bwd_dx(d, inter, sp);
        let mut bwd_b = rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_b,
                &["@model_path/weights/RW.bin"],
                &[&blob_rw[..]],
                &[blob_rw.len()],
                &[inter * sp * 2, inter * sp * 2],
                &[d * sp * 2],
            )
            .expect("bwd_b compile");

        FfnLayer {
            wg,
            wu,
            wd,
            d,
            inter,
            sp,
            fwd,
            bwd_a,
            bwd_b,
            dy16: vec![],
            gate: vec![],
            up: vec![],
        }
    }

    /// ANE forward pass: x → (gate, up, y)
    fn forward_ane(&mut self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x16 = to_fp16(x);
        self.fwd.write_input(0, &x16).expect("w");
        self.fwd.eval().expect("e");
        let gate = from_fp16(&self.fwd.read_output_vec(0).expect("r"));
        let up = from_fp16(&self.fwd.read_output_vec(1).expect("r"));
        let y = from_fp16(&self.fwd.read_output_vec(2).expect("r"));
        // Cache for backward
        self.gate = gate.clone();
        self.up = up.clone();
        (gate, up, y)
    }

    /// CPU forward pass: x → (gate, up, y)
    fn forward_cpu(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let gate = mm(&self.wg, self.inter, self.d, x, self.sp);
        let up = mm(&self.wu, self.inter, self.d, x, self.sp);
        let silu: Vec<f32> = gate
            .iter()
            .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
            .collect();
        let fused: Vec<f32> = silu.iter().zip(up.iter()).map(|(&a, &b)| a * b).collect();
        let down = mm(&self.wd, self.d, self.inter, &fused, self.sp);
        let mut y = vec![0.0f32; self.d * self.sp];
        for i in 0..y.len() {
            y[i] = down[i] + x[i];
        }
        (gate, up, y)
    }

    /// ANE-assisted backward: returns (dx, dL/dWg, dL/dWu, dL/dWd)
    fn backward_hybrid(
        &mut self,
        dy: &[f32],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d;
        let inter = self.inter;
        let sp = self.sp;

        // Step 1: ANE — dfused = Wd^T @ dy
        let dy16 = to_fp16(dy);
        self.bwd_a.write_input(0, &dy16).expect("w");
        self.bwd_a.eval().expect("e");
        let dfused = from_fp16(&self.bwd_a.read_output_vec(0).expect("r"));

        // Step 2: CPU — SiLU' + dgate + dup
        let silu: Vec<f32> = self
            .gate
            .iter()
            .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
            .collect();
        let dsilu: Vec<f32> = dfused
            .iter()
            .zip(self.up.iter())
            .map(|(&d, &u)| d * u)
            .collect();
        let dup: Vec<f32> = dfused
            .iter()
            .zip(silu.iter())
            .map(|(&d, &s)| d * s)
            .collect();
        let dgate: Vec<f32> = self
            .gate
            .iter()
            .zip(dsilu.iter())
            .map(|(&g, &ds)| {
                let s = 1.0 / (1.0 + (-g).exp());
                ds * s * (1.0 + g * (1.0 - s))
            })
            .collect();

        // Step 3: ANE — dx_partial = [WgT|WuT] @ concat(dgate, dup)
        let dgate16 = to_fp16(&dgate);
        let dup16 = to_fp16(&dup);
        self.bwd_b.write_input(0, &dgate16).expect("w");
        self.bwd_b.write_input(1, &dup16).expect("w");
        self.bwd_b.eval().expect("e");
        let dx_partial = from_fp16(&self.bwd_b.read_output_vec(0).expect("r"));

        // Step 4: CPU — dx = dx_partial + dy
        let mut dx = vec![0.0f32; d * sp];
        for i in 0..dx.len() {
            dx[i] = dx_partial[i] + dy[i];
        }

        // Step 5: CPU — weight gradients
        let dwg = mm_abt(&dgate, inter, sp, x, d);
        let dwu = mm_abt(&dup, inter, sp, x, d);
        let fused: Vec<f32> = self
            .gate
            .iter()
            .zip(self.up.iter())
            .map(|(&g, &u)| {
                let s = 1.0 / (1.0 + (-g).exp());
                g * s * u
            })
            .collect();
        let dwd = mm_abt(dy, d, sp, &fused, inter);

        (dx, dwg, dwu, dwd)
    }

    /// Pure CPU backward: returns (dx, dL/dWg, dL/dWu, dL/dWd)
    fn backward_cpu(&self, dy: &[f32], x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d;
        let inter = self.inter;
        let sp = self.sp;

        let dfused = mm_at(&self.wd, d, inter, dy, sp);
        let silu: Vec<f32> = self
            .gate
            .iter()
            .map(|&g| g * (1.0 / (1.0 + (-g).exp())))
            .collect();
        let dup: Vec<f32> = dfused
            .iter()
            .zip(silu.iter())
            .map(|(&d, &s)| d * s)
            .collect();
        let dsilu: Vec<f32> = dfused
            .iter()
            .zip(self.up.iter())
            .map(|(&d, &u)| d * u)
            .collect();
        let dgate: Vec<f32> = self
            .gate
            .iter()
            .zip(dsilu.iter())
            .map(|(&g, &ds)| {
                let s = 1.0 / (1.0 + (-g).exp());
                ds * s * (1.0 + g * (1.0 - s))
            })
            .collect();
        let dx_g = mm_at(&self.wg, inter, d, &dgate, sp);
        let dx_u = mm_at(&self.wu, inter, d, &dup, sp);
        let mut dx = vec![0.0f32; d * sp];
        for i in 0..dx.len() {
            dx[i] = dx_g[i] + dx_u[i] + dy[i];
        }

        let fused: Vec<f32> = silu
            .iter()
            .zip(self.up.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let dwg = mm_abt(&dgate, inter, sp, x, d);
        let dwu = mm_abt(&dup, inter, sp, x, d);
        let dwd = mm_abt(dy, d, sp, &fused, inter);

        (dx, dwg, dwu, dwd)
    }

    /// SGD weight update
    fn sgd_update(&mut self, dwg: &[f32], dwu: &[f32], dwd: &[f32], lr: f32) {
        let n = self.wg.len();
        unsafe {
            cblas_saxpy(n as i32, -lr, dwg.as_ptr(), 1, self.wg.as_mut_ptr(), 1);
            cblas_saxpy(n as i32, -lr, dwu.as_ptr(), 1, self.wu.as_mut_ptr(), 1);
        }
        let n = self.wd.len();
        unsafe {
            cblas_saxpy(n as i32, -lr, dwd.as_ptr(), 1, self.wd.as_mut_ptr(), 1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(768);
    let sp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let layers: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let steps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
    let inter = d * 4;
    let lr = 0.001_f32;

    println!("============================================================");
    println!("  Hybrid Training Step Benchmark");
    println!(
        "  D={} SP={} inter={} layers={} steps={}",
        d, sp, inter, layers, steps
    );
    println!("============================================================\n");

    // Create layers
    let mut ane_layers: Vec<FfnLayer> = (0..layers)
        .map(|i| FfnLayer::new(d, inter, sp, 42 + i as u64 * 1000))
        .collect();

    // Input data
    let x: Vec<f32> = rand_m(d, sp, 0.1, 9999);
    // Simple target: random projection
    let target_w = rand_m(d, d, 0.01, 12345);
    let target_out = mm(&target_w, d, d, &x, sp);

    // ── Warmup ──
    let mut tmp_x = x.clone();
    for layer in &mut ane_layers {
        let (gate, up, _y) = layer.forward_ane(&tmp_x);
        layer.gate = gate;
        layer.up = up;
        let dy = &vec![0.0f32; d * sp];
        let _ = layer.backward_hybrid(dy, &tmp_x);
    }

    // ── Benchmark: ANE Hybrid Training Step ──
    let mut hybrid_step_times = Vec::new();
    let mut hybrid_fwd_times = Vec::new();
    let mut hybrid_bwd_times = Vec::new();
    let mut loss_sum = 0.0_f64;

    for step in 0..steps {
        let mut tmp_x = x.clone();
        let mut layer_inputs: Vec<Vec<f32>> = Vec::new();
        layer_inputs.push(tmp_x.clone());

        let t_step = Instant::now();
        let t_fwd = Instant::now();

        // Forward pass through all layers
        for layer in &mut ane_layers {
            let (_gate, _up, y) = layer.forward_ane(&tmp_x);
            tmp_x = y;
            layer_inputs.push(tmp_x.clone());
        }

        let fwd_t = t_fwd.elapsed().as_secs_f64() * 1000.0;

        // MSE loss
        let final_y = &tmp_x;
        let n = final_y.len() as f32;
        let mut loss = 0.0_f32;
        let mut dy = vec![0.0f32; n as usize];
        for i in 0..final_y.len() {
            let diff = final_y[i] - target_out[i];
            loss += diff * diff;
            dy[i] = 2.0 * diff / n;
        }
        loss /= n as f32;
        loss_sum += loss as f64;

        // Backward pass
        let t_bwd = Instant::now();
        let mut grad = dy;

        for i in (0..layers).rev() {
            let (dx, dwg, dwu, dwd) = ane_layers[i].backward_hybrid(&grad, &layer_inputs[i]);
            ane_layers[i].sgd_update(&dwg, &dwu, &dwd, lr);
            grad = dx;
        }

        let bwd_t = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let step_t = t_step.elapsed().as_secs_f64() * 1000.0;

        hybrid_fwd_times.push(fwd_t);
        hybrid_bwd_times.push(bwd_t);
        hybrid_step_times.push(step_t);
    }

    let hybrid_step = hybrid_step_times.iter().sum::<f64>() / steps as f64;
    let hybrid_fwd = hybrid_fwd_times.iter().sum::<f64>() / steps as f64;
    let hybrid_bwd = hybrid_bwd_times.iter().sum::<f64>() / steps as f64;

    // ── Benchmark: Pure CPU Training Step ──
    // Recreate layers for CPU (fresh weights)
    let mut cpu_layers: Vec<FfnLayer> = (0..layers)
        .map(|i| FfnLayer::new(d, inter, sp, 42 + i as u64 * 1000))
        .collect();

    let mut cpu_step_times = Vec::new();
    let mut cpu_fwd_times = Vec::new();
    let mut cpu_bwd_times = Vec::new();

    for _step in 0..steps {
        let mut tmp_x = x.clone();
        let mut layer_inputs: Vec<Vec<f32>> = Vec::new();
        layer_inputs.push(tmp_x.clone());

        let t_step = Instant::now();
        let t_fwd = Instant::now();

        for layer in &mut cpu_layers {
            let (gate, up, y) = layer.forward_cpu(&tmp_x);
            layer.gate = gate;
            layer.up = up;
            tmp_x = y;
            layer_inputs.push(tmp_x.clone());
        }

        let fwd_t = t_fwd.elapsed().as_secs_f64() * 1000.0;

        let final_y = &tmp_x;
        let n = final_y.len() as f32;
        let mut dy = vec![0.0f32; n as usize];
        for i in 0..final_y.len() {
            let diff = final_y[i] - target_out[i];
            dy[i] = 2.0 * diff / n;
        }

        let t_bwd = Instant::now();
        let mut grad = dy;
        for i in (0..layers).rev() {
            let (dx, dwg, dwu, dwd) = cpu_layers[i].backward_cpu(&grad, &layer_inputs[i]);
            cpu_layers[i].sgd_update(&dwg, &dwu, &dwd, lr);
            grad = dx;
        }
        let bwd_t = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let step_t = t_step.elapsed().as_secs_f64() * 1000.0;

        cpu_fwd_times.push(fwd_t);
        cpu_bwd_times.push(bwd_t);
        cpu_step_times.push(step_t);
    }

    let cpu_step = cpu_step_times.iter().sum::<f64>() / steps as f64;
    let cpu_fwd = cpu_fwd_times.iter().sum::<f64>() / steps as f64;
    let cpu_bwd = cpu_bwd_times.iter().sum::<f64>() / steps as f64;

    // ── Results ──
    println!("=== Per-Step Timing ({} layers) ===\n", layers);
    println!(
        "  {:35} {:>10} {:>10} {:>10}",
        "", "Hybrid", "CPU", "Speedup"
    );
    println!(
        "  {:35} {:>10} {:>10} {:>10}",
        "─".repeat(35),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}x",
        "Forward",
        format!("{:.2}", hybrid_fwd),
        format!("{:.2}", cpu_fwd),
        format!("{:.2}", cpu_fwd / hybrid_fwd)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}x",
        "Backward (input + weight grads)",
        format!("{:.2}", hybrid_bwd),
        format!("{:.2}", cpu_bwd),
        format!("{:.2}", cpu_bwd / hybrid_bwd)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}x",
        "Total step",
        format!("{:.2}", hybrid_step),
        format!("{:.2}", cpu_step),
        format!("{:.2}", cpu_step / hybrid_step)
    );
    println!(
        "\n  Loss: {:.6} (avg over {} steps)",
        loss_sum / steps as f64,
        steps
    );
    println!(
        "\n  ANE compiles: {} per layer ({} forward + {} backward)",
        3, 1, 2
    );
}
