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
fn to_fp16_inplace(src: &[f32], dst: &mut [u8]) {
    for (i, &v) in src.iter().enumerate() {
        let h = f16::from_f32(v).to_bits();
        dst[i * 2] = (h & 0xFF) as u8;
        dst[i * 2 + 1] = (h >> 8) as u8;
    }
}
fn from_fp16_inplace(src: &[u8], dst: &mut [f32]) {
    for i in 0..dst.len() {
        let h = (src[i * 2] as u16) | ((src[i * 2 + 1] as u16) << 8);
        dst[i] = f16::from_bits(h).to_f32();
    }
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
    bwd_a: Option<rustane::wrapper::ANEExecutor>,
    bwd_b: Option<rustane::wrapper::ANEExecutor>,
    // Cached backward intermediates (CPU)
    gate: Vec<f32>,
    up: Vec<f32>,
    // Pre-allocated buffers to avoid per-call allocation
    // fp16 buffers for ANE I/O
    x16_buf: Vec<u8>,
    dy16_buf: Vec<u8>,
    dgate16_buf: Vec<u8>,
    dup16_buf: Vec<u8>,
    // fp16 output buffers (read directly from ANE)
    gate16_buf: Vec<u8>,
    up16_buf: Vec<u8>,
    y16_buf: Vec<u8>,
    dfused16_buf: Vec<u8>,
    dx16_buf: Vec<u8>,
    y_buf: Vec<f32>,
    // fp32 intermediate buffers (backward)
    dfused_buf: Vec<f32>,
    silu_buf: Vec<f32>,
    dsilu_buf: Vec<f32>,
    dup_buf: Vec<f32>,
    dgate_buf: Vec<f32>,
    dx_buf: Vec<f32>,
    fused_buf: Vec<f32>,
}

impl FfnLayer {
    /// Full layer with forward + backward ANE programs (3 compiles)
    fn new(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        let (wg, wu, wd, wdi, wdt, rw) = Self::compute_weights(d, inter, seed);
        let fwd = Self::compile_forward(d, inter, sp, &wg, &wu, &wdi);
        let bwd_a = Some(Self::compile_bwd_a(d, inter, sp, &wdt));
        let bwd_b = Some(Self::compile_bwd_b(d, inter, sp, &rw));
        let mut layer = Self::allocate(d, inter, sp, wg, wu, wd, fwd);
        layer.bwd_a = bwd_a;
        layer.bwd_b = bwd_b;
        layer
    }

    /// Forward-only layer (1 ANE compile) — backward uses CPU only
    fn new_forward_only(d: usize, inter: usize, sp: usize, seed: u64) -> Self {
        let (wg, wu, wd, wdi, _wdt, _rw) = Self::compute_weights(d, inter, seed);
        let fwd = Self::compile_forward(d, inter, sp, &wg, &wu, &wdi);
        Self::allocate(d, inter, sp, wg, wu, wd, fwd)
    }

    fn compute_weights(
        d: usize,
        inter: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let wg = rand_m(inter, d, 0.02, seed);
        let wu = rand_m(inter, d, 0.02, seed + 1);
        let wd = rand_m(d, inter, 0.02, seed + 2);

        // Forward weight: WdI = [Wd | I] concatenated along input channels
        let mut wdi = vec![0.0f32; d * (inter + d)];
        for r in 0..d {
            for c in 0..inter {
                wdi[r * (inter + d) + c] = wd[r * inter + c];
            }
            wdi[r * (inter + d) + inter + r] = 1.0;
        }

        // Backward weights
        let mut wdt = vec![0.0f32; inter * d];
        for i in 0..inter {
            for j in 0..d {
                wdt[i * d + j] = wd[j * inter + i];
            }
        }
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

        (wg, wu, wd, wdi, wdt, rw)
    }

    fn compile_forward(
        d: usize,
        inter: usize,
        sp: usize,
        wg: &[f32],
        wu: &[f32],
        wdi: &[f32],
    ) -> rustane::wrapper::ANEExecutor {
        let blob_wg = build_blob(wg);
        let blob_wu = build_blob(wu);
        let blob_wdi = build_blob(wdi);
        let mil_fwd = mil_forward(d, inter, sp);
        rustane::wrapper::ANECompiler::new()
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
            .expect("fwd compile")
    }

    fn compile_bwd_a(
        d: usize,
        inter: usize,
        sp: usize,
        wdt: &[f32],
    ) -> rustane::wrapper::ANEExecutor {
        let blob_wdt = build_blob(wdt);
        let mil_a = mil_bwd_dfused(d, inter, sp);
        rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_a,
                &["@model_path/weights/WdT.bin"],
                &[&blob_wdt[..]],
                &[blob_wdt.len()],
                &[d * sp * 2],
                &[inter * sp * 2],
            )
            .expect("bwd_a compile")
    }

    fn compile_bwd_b(
        d: usize,
        inter: usize,
        sp: usize,
        rw: &[f32],
    ) -> rustane::wrapper::ANEExecutor {
        let blob_rw = build_blob(rw);
        let mil_b = mil_bwd_dx(d, inter, sp);
        rustane::wrapper::ANECompiler::new()
            .compile_multi(
                &mil_b,
                &["@model_path/weights/RW.bin"],
                &[&blob_rw[..]],
                &[blob_rw.len()],
                &[inter * sp * 2, inter * sp * 2],
                &[d * sp * 2],
            )
            .expect("bwd_b compile")
    }

    fn allocate(
        d: usize,
        inter: usize,
        sp: usize,
        wg: Vec<f32>,
        wu: Vec<f32>,
        wd: Vec<f32>,
        fwd: rustane::wrapper::ANEExecutor,
    ) -> Self {
        let x16_buf = vec![0u8; d * sp * 2];
        let dy16_buf = vec![0u8; d * sp * 2];
        let dgate16_buf = vec![0u8; inter * sp * 2];
        let dup16_buf = vec![0u8; inter * sp * 2];
        let gate16_buf = vec![0u8; inter * sp * 2];
        let up16_buf = vec![0u8; inter * sp * 2];
        let y16_buf = vec![0u8; d * sp * 2];
        let dfused16_buf = vec![0u8; inter * sp * 2];
        let dx16_buf = vec![0u8; d * sp * 2];
        let y_buf = vec![0.0f32; d * sp];
        let dfused_buf = vec![0.0f32; inter * sp];
        let silu_buf = vec![0.0f32; inter * sp];
        let dsilu_buf = vec![0.0f32; inter * sp];
        let dup_buf = vec![0.0f32; inter * sp];
        let dgate_buf = vec![0.0f32; inter * sp];
        let dx_buf = vec![0.0f32; d * sp];
        let fused_buf = vec![0.0f32; inter * sp];

        FfnLayer {
            wg,
            wu,
            wd,
            d,
            inter,
            sp,
            fwd,
            bwd_a: None,
            bwd_b: None,
            gate: vec![0.0f32; inter * sp],
            up: vec![0.0f32; inter * sp],
            x16_buf,
            dy16_buf,
            dgate16_buf,
            dup16_buf,
            gate16_buf,
            up16_buf,
            y16_buf,
            dfused16_buf,
            dx16_buf,
            y_buf,
            dfused_buf,
            silu_buf,
            dsilu_buf,
            dup_buf,
            dgate_buf,
            dx_buf,
            fused_buf,
        }
    }

    /// ANE forward pass: x → (gate, up, y) using pre-allocated buffers
    fn forward_ane(&mut self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Convert x to fp16 in-place into pre-allocated buffer
        to_fp16_inplace(x, &mut self.x16_buf);
        self.fwd.write_input(0, &self.x16_buf).expect("w");
        self.fwd.eval().expect("e");
        // Read outputs into pre-allocated buffers
        self.fwd.read_output(0, &mut self.gate16_buf).expect("r");
        self.fwd.read_output(1, &mut self.up16_buf).expect("r");
        self.fwd.read_output(2, &mut self.y16_buf).expect("r");
        // Convert fp16 → fp32 in-place
        from_fp16_inplace(&self.gate16_buf, &mut self.gate);
        from_fp16_inplace(&self.up16_buf, &mut self.up);
        from_fp16_inplace(&self.y16_buf, &mut self.y_buf);
        (self.gate.clone(), self.up.clone(), self.y_buf.clone())
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
    /// Uses pre-allocated buffers to minimize allocation overhead
    /// Only works if backward executors were compiled (bwd_a/bwd_b = Some)
    fn backward_hybrid(
        &mut self,
        dy: &[f32],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let d = self.d;
        let inter = self.inter;
        let sp = self.sp;
        let bwd_a = self
            .bwd_a
            .as_mut()
            .expect("backward_hybrid requires compiled backward executors");
        let bwd_b = self
            .bwd_b
            .as_mut()
            .expect("backward_hybrid requires compiled backward executors");

        // Step 1: ANE — dfused = Wd^T @ dy
        to_fp16_inplace(dy, &mut self.dy16_buf);
        bwd_a.write_input(0, &self.dy16_buf).expect("w");
        bwd_a.eval().expect("e");
        bwd_a.read_output(0, &mut self.dfused16_buf).expect("r");
        from_fp16_inplace(&self.dfused16_buf, &mut self.dfused_buf);

        // Step 2: CPU — SiLU' + dgate + dup (in-place into pre-allocated buffers)
        // silu_buf = gate * sigmoid(gate)
        for i in 0..inter * sp {
            let g = self.gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            self.silu_buf[i] = g * s;
        }
        // dsilu = dfused * up
        for i in 0..inter * sp {
            self.dsilu_buf[i] = self.dfused_buf[i] * self.up[i];
        }
        // dup = dfused * silu
        for i in 0..inter * sp {
            self.dup_buf[i] = self.dfused_buf[i] * self.silu_buf[i];
        }
        // dgate = dsilu * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        for i in 0..inter * sp {
            let g = self.gate[i];
            let s = 1.0 / (1.0 + (-g).exp());
            self.dgate_buf[i] = self.dsilu_buf[i] * s * (1.0 + g * (1.0 - s));
        }

        // Step 3: ANE — dx_partial = [WgT|WuT] @ concat(dgate, dup)
        to_fp16_inplace(&self.dgate_buf, &mut self.dgate16_buf);
        to_fp16_inplace(&self.dup_buf, &mut self.dup16_buf);
        bwd_b.write_input(0, &self.dgate16_buf).expect("w");
        bwd_b.write_input(1, &self.dup16_buf).expect("w");
        bwd_b.eval().expect("e");
        bwd_b.read_output(0, &mut self.dx16_buf).expect("r");
        from_fp16_inplace(&self.dx16_buf, &mut self.dx_buf);

        // Step 4: CPU — dx = dx_partial + dy
        for i in 0..d * sp {
            self.dx_buf[i] += dy[i];
        }

        // Step 5: CPU — weight gradients (into pre-allocated buffers)
        // fused = silu * up (reuse dsilu_buf as temp — compute fused first)
        for i in 0..inter * sp {
            self.fused_buf[i] = self.silu_buf[i] * self.up[i];
        }
        // dWg = dgate^T @ x^T → [inter, sp]^T @ [sp, d] = [inter, d]
        let wg = mm_abt(&self.dgate_buf, inter, sp, x, d);
        let wu = mm_abt(&self.dup_buf, inter, sp, x, d);
        let wd = mm_abt(dy, d, sp, &self.fused_buf, inter);

        (self.dx_buf.clone(), wg, wu, wd)
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
        let (_gate, _up, _y) = layer.forward_ane(&tmp_x);
        // gate/up already cached by forward_ane
        let dy = vec![0.0f32; d * sp];
        let _ = layer.backward_hybrid(&dy, &tmp_x);
    }

    // ── Benchmark: ANE Hybrid Training Step ──
    let mut hybrid_step_times = Vec::new();
    let mut hybrid_fwd_times = Vec::new();
    let mut hybrid_bwd_times = Vec::new();
    let mut loss_sum = 0.0_f64;

    for _step in 0..steps {
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

    // ── Benchmark: ANE Forward Only + CPU Backward ──
    // This avoids the ANE backward overhead while keeping the forward speedup
    let mut fwd_only_layers: Vec<FfnLayer> = (0..layers)
        .map(|i| FfnLayer::new_forward_only(d, inter, sp, 42 + i as u64 * 1000))
        .collect();

    let mut fwd_only_step_times = Vec::new();
    let mut fwd_only_fwd_times = Vec::new();
    let mut fwd_only_bwd_times = Vec::new();

    for _step in 0..steps {
        let mut tmp_x = x.clone();
        let mut layer_inputs: Vec<Vec<f32>> = Vec::new();
        layer_inputs.push(tmp_x.clone());

        let t_step = Instant::now();
        let t_fwd = Instant::now();

        // ANE forward
        for layer in &mut fwd_only_layers {
            let (_gate, _up, y) = layer.forward_ane(&tmp_x);
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

        // CPU backward
        let t_bwd = Instant::now();
        let mut grad = dy;
        for i in (0..layers).rev() {
            let (dx, dwg, dwu, dwd) = fwd_only_layers[i].backward_cpu(&grad, &layer_inputs[i]);
            fwd_only_layers[i].sgd_update(&dwg, &dwu, &dwd, lr);
            grad = dx;
        }
        let bwd_t = t_bwd.elapsed().as_secs_f64() * 1000.0;
        let step_t = t_step.elapsed().as_secs_f64() * 1000.0;

        fwd_only_fwd_times.push(fwd_t);
        fwd_only_bwd_times.push(bwd_t);
        fwd_only_step_times.push(step_t);
    }

    let fwd_only_step = fwd_only_step_times.iter().sum::<f64>() / steps as f64;
    let fwd_only_fwd = fwd_only_fwd_times.iter().sum::<f64>() / steps as f64;
    let fwd_only_bwd = fwd_only_bwd_times.iter().sum::<f64>() / steps as f64;

    // ── Results ──
    println!("=== Per-Step Timing ({} layers) ===\n", layers);
    println!(
        "  {:35} {:>10} {:>10} {:>10} {:>10}",
        "", "ANE Fwd", "Hybrid", "CPU", "Fwd+CPU"
    );
    println!(
        "  {:35} {:>10} {:>10} {:>10} {:>10}",
        "─".repeat(35),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}ms {:>8}ms",
        "Forward",
        format!("{:.2}", fwd_only_fwd),
        format!("{:.2}", hybrid_fwd),
        format!("{:.2}", cpu_fwd),
        format!("{:.2}", fwd_only_fwd + fwd_only_bwd)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}ms",
        "Backward",
        "-",
        format!("{:.2}", hybrid_bwd),
        format!("{:.2}", cpu_bwd)
    );
    println!(
        "  {:35} {:>8}ms {:>8}ms {:>8}ms",
        "Total step",
        format!("{:.2}", fwd_only_step),
        format!("{:.2}", hybrid_step),
        format!("{:.2}", cpu_step)
    );
    println!();
    println!("  Speedup vs CPU:");
    println!(
        "  {:35} {:>10} {:>10} {:>10}",
        "", "ANE Fwd", "Hybrid", "Fwd+CPU"
    );
    println!(
        "  {:35} {:>10} {:>10} {:>10}",
        "─".repeat(35),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(10)
    );
    println!(
        "  {:35} {:>8}x {:>8}x {:>8}x",
        "Forward",
        format!("{:.2}", cpu_fwd / fwd_only_fwd),
        format!("{:.2}", cpu_fwd / hybrid_fwd),
        "-"
    );
    println!(
        "  {:35} {:>8}x {:>8}x {:>8}x",
        "Backward",
        "-",
        format!("{:.2}", cpu_bwd / hybrid_bwd),
        "-"
    );
    println!(
        "  {:35} {:>8}x {:>8}x {:>8}x",
        "Total step",
        "-",
        format!("{:.2}", cpu_step / hybrid_step),
        format!("{:.2}", cpu_step / fwd_only_step)
    );
    println!(
        "\n  Loss: {:.6} (avg over {} steps)",
        loss_sum / steps as f64,
        steps
    );
    println!(
        "\n  ANE compiles: {} per layer (forward only) vs {} per layer (hybrid)",
        1, 3
    );

    // Strategy recommendation
    let _best_step = hybrid_step.min(fwd_only_step).min(cpu_step);
    println!("\n  Recommendation: ",);
    if fwd_only_step <= hybrid_step && fwd_only_step < cpu_step {
        println!("    → ANE forward + CPU backward (saves compile budget, best total time)");
    } else if hybrid_step <= fwd_only_step && hybrid_step < cpu_step {
        println!("    → Full hybrid (ANE forward + backward)");
    } else {
        println!("    → Pure CPU (ANE overhead exceeds compute savings)");
    }
}
