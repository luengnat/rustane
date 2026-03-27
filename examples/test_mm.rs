#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        l: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        b: f32,
        C: *mut f32,
        ldc: i32,
    );
}

fn mm_nn(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
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
// C = A^T @ B, A stored [k,m], B stored [k,n], C is [m,n]
fn mm_nt(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        cblas_sgemm(
            101,
            112,
            111,
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

fn main() {
    // Test: A = [[1,2],[3,4],[5,6]] (3x2), B = [[7,8],[9,10]] (2x2)
    // A^T = [[1,3,5],[2,4,6]] (2x3)
    // A^T @ B = [[1*7+3*9+5*9, 1*8+3*10+5*?], ...] -- let me use simpler
    // A = [[1,2],[3,4]] (2x3 row-major = [1,2,3,4]), B = [[5],[6],[7]] (3x1)
    // A^T = [[1,3],[2,4]] (3x2), A^T @ B = [[1*5+3*6+?]] -- hmm

    // Simpler: A = [1,2; 3,4] (2x2, row-major [1,2,3,4])
    // A^T = [1,3; 2,4]
    // B = [5,6; 7,8] (2x2)
    // A^T @ B = [1*5+3*7, 1*6+3*8; 2*5+4*7, 2*6+4*8] = [26, 30; 38, 44]

    let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [2,2] row-major
    let b = vec![5.0f32, 6.0, 7.0, 8.0]; // [2,2] row-major

    // mm_nt(m=2, n=2, k=2, a=[2,2], b=[2,2])
    // Should compute A^T @ B where A is [2,2], B is [2,2]
    // Result should be [26, 30, 38, 44]
    let c = mm_nt(2, 2, 2, &a, &b);
    println!("mm_nt(2,2,2): {:?}", c);
    println!("Expected: [26, 30, 38, 44]");

    // Now test the gradient case: W=[m,k]=[4,3], dy=[m,n]=[4,2]
    // Want dx = W^T @ dy, where W^T is [k,m]=[3,4], result is [k,n]=[3,2]
    // Call: mm_nt(m=k=3, n=2, k=m=4, &w, &dy)
    // Inside: BLAS M=3, N=2, K=4, op(A)=A^T, A stored [4,3], lda=3
    // op(A)=[3,4], B=[4,2], C=[3,2] ✓
    let w = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ]; // [4,3]
    let dy = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]; // [4,2]
                                                           // W^T = [[1,4,7,10],[2,5,8,11],[3,6,9,12]]
                                                           // W^T @ dy = [[1*1+4*0+7*0+10*1, 1*0+4*1+7*0+10*0], [2,5,8,11]@dy, [3,6,9,12]@dy]
                                                           //        = [[11, 4], [2, 5], [3, 6]]
    let dx = mm_nt(3, 2, 4, &w, &dy);
    println!("mm_nt(3,2,4): {:?}", dx);
    println!("Expected: [11, 4, 2, 5, 3, 6]");

    // Manual computation
    let mut dx2 = vec![0.0f32; 3 * 2];
    for i in 0..3 {
        for j in 0..2 {
            dx2[i * 2 + j] = 0.0;
            for kk in 0..4 {
                dx2[i * 2 + j] += w[kk * 3 + i] * dy[kk * 2 + j];
            }
        }
    }
    println!("Manual: {:?}", dx2);
}
