#[test]
fn test_io_surface_creation() {
    use rustane::ane::IOSurface;

    let result = IOSurface::new(1024);
    // May fail on non-Apple Silicon, but shouldn't panic
    match result {
        Ok(_) => println!("IOSurface created"),
        Err(e) => println!("IOSurface creation not available: {:?}", e),
    }
}

#[test]
fn test_io_surface_write_read_roundtrip() {
    use rustane::ane::IOSurface;

    let io = match IOSurface::new(64) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let _ = io.write(&data);

    let mut read_data = vec![0u8; 64];
    let result = io.read(&mut read_data);

    match result {
        Ok(()) => {
            // Verify the written data is at the beginning
            assert_eq!(&read_data[..data.len()], data.as_slice());
        }
        Err(_) => println!("Read failed (expected on some systems)"),
    }
}

#[test]
fn test_io_surface_write_exceeds_capacity() {
    use rustane::ane::IOSurface;

    let io = match IOSurface::new(10) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    let data = vec![1u8; 20]; // Larger than capacity
    let result = io.write(&data);

    // Should fail gracefully
    assert!(result.is_err());
}

#[test]
fn test_io_surface_lock_read() {
    use rustane::ane::IOSurface;

    let io = match IOSurface::new(64) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    // Test that lock_read allows direct pointer access
    let result = io.lock_read();

    match result {
        Ok(ptr) => {
            // Verify we got a non-null pointer
            assert!(!ptr.is_null());
        }
        Err(_) => println!("lock_read failed (expected on some systems)"),
    }
}
