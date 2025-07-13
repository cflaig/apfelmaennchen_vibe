// Mandelbrot set compute shader
// This shader calculates the Mandelbrot set on the GPU

// Uniform buffer containing parameters for the Mandelbrot calculation
struct Params {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    max_iterations: u32,
    width: u32,
    height: u32,
    color_offset: f32,
    formula: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// Helper function to convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> u32 {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - abs(fract(h_prime / 2.0) * 2.0 - 1.0));
    let m = v - c;
    
    var r: f32;
    var g: f32;
    var b: f32;
    
    if (h_prime < 1.0) {
        r = c; g = x; b = 0.0;
    } else if (h_prime < 2.0) {
        r = x; g = c; b = 0.0;
    } else if (h_prime < 3.0) {
        r = 0.0; g = c; b = x;
    } else if (h_prime < 4.0) {
        r = 0.0; g = x; b = c;
    } else if (h_prime < 5.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    
    let r_u8 = u32(255.0 * (r + m));
    let g_u8 = u32(255.0 * (g + m));
    let b_u8 = u32(255.0 * (b + m));
    
    return (255u << 24u) | (b_u8 << 16u) | (g_u8 << 8u) | r_u8;
}

// Helper function to perform one iteration of the Mandelbrot formula
fn iterate_point(z_re: f32, z_im: f32, c_re: f32, c_im: f32, formula: u32) -> vec2<f32> {
    if (formula == 0u) {
        // Standard formula: Z_n = Z_n-1^2 + C
        let new_re = z_re * z_re - z_im * z_im + c_re;
        let new_im = 2.0 * z_re * z_im + c_im;
        return vec2<f32>(new_re, new_im);
    } else {
        // Alternative formula: Z_n = Z_n-1^2 - i*Z_n-1 + C
        let new_re = z_re * z_re - z_im * z_im - z_im + c_re;
        let new_im = 2.0 * z_re * z_im - z_re + c_im;
        return vec2<f32>(new_re, new_im);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Check if we're within the image bounds
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // Convert pixel coordinates to complex plane coordinates
    let aspect_ratio = f32(params.width) / f32(params.height);
    let scale_x = params.zoom * aspect_ratio;
    let scale_y = params.zoom;
    
    let real = params.center_x + (f32(x) / f32(params.width) - 0.5) * scale_x;
    let imag = params.center_y + (f32(y) / f32(params.height) - 0.5) * scale_y;
    
    // Mandelbrot iteration
    var z_re: f32 = 0.0;
    var z_im: f32 = 0.0;
    let c_re = real;
    let c_im = imag;
    
    let escape_radius_sq: f32 = 256.0; // 16.0^2
    var i: u32 = 0u;
    var smooth_i: f32 = 0.0;
    
    for (; i < params.max_iterations; i = i + 1u) {
        // Perform iteration
        let z = iterate_point(z_re, z_im, c_re, c_im, params.formula);
        z_re = z.x;
        z_im = z.y;
        
        // Check if point escaped
        let mag_sq = z_re * z_re + z_im * z_im;
        if (mag_sq > escape_radius_sq) {
            // Calculate smooth iteration count for better coloring
            smooth_i = f32(i) + 1.0 - log(log(mag_sq) / log(escape_radius_sq)) / log(2.0);
            break;
        }
    }
    
    // Calculate color
    var color: u32;
    if (i == params.max_iterations) {
        // Point is in the set - black
        color = 0xFF000000u; // RGBA: Black with full alpha
    } else {
        // Point is outside the set - use smooth coloring
        let t = smooth_i / f32(params.max_iterations);
        
        // Create a rainbow-like effect with smooth transitions
        let hue = (0.95 * t * 360.0 + params.color_offset) % 360.0;
        color = hsv_to_rgb(hue, 0.8, 1.0);
    }
    
    // Write the color to the output buffer
    let index = y * params.width + x;
    output[index] = color;
}