use std::time::{Duration, Instant};
use env_logger::Env;
use log::error;
use num_complex::{Complex, Complex64};
use pixels::{Error, Pixels, SurfaceTexture};
use rayon::prelude::*;
use winit::dpi::LogicalSize;
use winit::event::VirtualKeyCode;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder};
use winit_input_helper::WinitInputHelper;

mod gpu_mandelbrot;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const MAX_ITERATIONS: u32 = 51200;
const ESCAPE_RADIUS_SQ: f64 = 256.0; // 16.0^2

struct Mandelbrot {
    // View parameters
    center_x: f64,
    center_y: f64,
    zoom: f64,
    // Color parameters
    color_offset: f64,
    // Current iteration level for progressive rendering
    current_iterations: u32,
    max_iterations: u32,
    // Previous view parameters to detect changes
    prev_center_x: f64,
    prev_center_y: f64,
    prev_zoom: f64,
    prev_formula: u8,
    // Mouse tracking
    mouse_pos: Option<(f64, f64)>,
    // Orbit of the point under the mouse cursor
    orbit: Vec<(f64, f64)>,
    // Whether the orbit converges (true) or diverges (false)
    orbit_converges: bool,
    // Formula selection (0 = standard, 1 = alternative)
    formula: u8,
    // Window dimensions
    width: u32,
    height: u32,
    cache: Vec<(u32, f64, Complex64)>,
    // GPU implementation
    gpu_mandelbrot: Option<gpu_mandelbrot::GpuMandelbrot>,
    // Flag to use GPU implementation
    use_gpu: bool
}

impl Mandelbrot {
    fn new() -> Self {
        // Initialize GPU implementation
        let gpu_mandelbrot = pollster::block_on(async {
            gpu_mandelbrot::GpuMandelbrot::new(WIDTH, HEIGHT).await
        });

        Self {
            center_x: -0.5,
            center_y: 0.0,
            zoom: 4.0,
            color_offset: 0.0,
            current_iterations: 2, // Start with 1 iteration
            max_iterations: 51200,
            prev_center_x: -0.5,
            prev_center_y: 0.0,
            prev_zoom: 4.0,
            prev_formula: 0,
            mouse_pos: None,
            orbit: Vec::new(),
            orbit_converges: false,
            formula: 0, // 0 = standard formula, 1 = alternative formula
            width: WIDTH,
            height: HEIGHT,
            cache: vec![(0, 0.0, Complex64::new(0.0, 0.0)); (WIDTH * HEIGHT) as usize],
            gpu_mandelbrot: Some(gpu_mandelbrot),
            use_gpu: true // Use GPU by default
        }
    }

    fn cycle_colors(&mut self, amount: f64) {
        self.color_offset = (self.color_offset + amount) % 360.0;
    }

    // Helper function to perform one iteration of the fixed-point formula using Complex numbers
    fn iterate_point(&self, z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
        if self.formula == 0 {
            // Standard formula: Z_n = Z_n-1^2 + C
            z * z + c
        } else {
            // Alternative formula: Z_n = Z_n-1^2 - i*Z_n-1 + C
            //z * z - Complex64::i() * z + Complex64::I/c
            z * z - Complex64::i() * z + c
        }
    }

    fn calculate(&self, x: u32, y: u32, iterations: u32, max_iterations: u32, smooth_it: f64, z: Complex64) -> (u32, f64, Complex64) {
        // Convert pixel coordinates to complex plane coordinates
        let width = self.width as f64;
        let height = self.height as f64;

        // Scale to make the view square and centered
        let aspect_ratio = width / height;
        let scale_x = self.zoom * aspect_ratio;
        let scale_y = self.zoom;

        let real = self.center_x + (x as f64 / width - 0.5) * scale_x;
        let imag = self.center_y + (y as f64 / height - 0.5) * scale_y;

        // Create complex number c
        let c = Complex::new(real, imag);

        // Mandelbrot iteration with Complex numbers
        let mut z = z;

        // Early bailout check
        let mag_sq = z.norm_sqr();
        if mag_sq > ESCAPE_RADIUS_SQ {
            return (iterations, iterations as f64, z);
        }

        // Use current_iterations for the calculation instead of max_iterations
        // This allows for progressive rendering
        for i in iterations..max_iterations {
            // Optimized iteration step using the helper function with Complex numbers
            z = self.iterate_point(z, c);

            // Early bailout check
            let mag_sq = z.norm_sqr();
            if mag_sq > ESCAPE_RADIUS_SQ {
                // Point escaped - calculate smooth iteration count for better coloring
                let smooth_i = i as f64 + 1.0 - (mag_sq.ln() / ESCAPE_RADIUS_SQ.ln()).ln() / 2.0_f64.ln();
                return (i, i as f64, z);
            }
        }

        // Point did not escape - in the set
        (max_iterations, 0.0, z)
    }

    // Helper method to draw a pixel
    fn draw_pixel(&self, frame: &mut [u8], x: u32, y: u32, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let pixel_offset = ((y * self.width + x) as usize) * 4;
            if pixel_offset + 3 < frame.len() {
                frame[pixel_offset..pixel_offset + 4].copy_from_slice(&color);
            }
        }
    }

    // Helper method to draw a line using Bresenham's algorithm
    fn draw_line(&self, frame: &mut [u8], x0: u32, y0: u32, x1: u32, y1: u32, color: [u8; 4]) {
        let mut x0 = x0 as i32;
        let mut y0 = y0 as i32;
        let x1 = x1 as i32;
        let y1 = y1 as i32;

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x0 >= 0 && y0 >= 0 && x0 < self.width as i32 && y0 < self.height as i32 {
                self.draw_pixel(frame, x0 as u32, y0 as u32, color);
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                if x0 == x1 {
                    break;
                }
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                if y0 == y1 {
                    break;
                }
                err += dx;
                y0 += sy;
            }
        }
    }

    // Helper method to draw a character (simple 5x7 bitmap font)
    fn draw_char(&self, frame: &mut [u8], x: u32, y: u32, c: char, color: [u8; 4]) {
        // Define a simple 5x7 bitmap font for ASCII characters
        // Each character is represented by 7 bytes, each byte represents a row
        // Each bit in the byte represents a pixel (1 = pixel, 0 = no pixel)
        let font = match c {
            '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
            '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
            '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
            '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
            '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
            '5' => [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b10001, 0b01110],
            '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
            '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
            '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
            '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
            'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
            'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
            'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
            'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
            'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
            'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
            'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
            'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
            'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
            'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
            'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
            'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
            'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
            'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
            'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
            'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
            'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
            'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
            'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
            'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
            'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
            'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
            'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
            'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
            'Y' => [0b10001, 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100],
            'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
            'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
            'b' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b11110],
            'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
            'd' => [0b00001, 0b00001, 0b01101, 0b10011, 0b10001, 0b10001, 0b01111],
            'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
            'f' => [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
            'g' => [0b00000, 0b00000, 0b01111, 0b10001, 0b01111, 0b00001, 0b01110],
            'h' => [0b10000, 0b10000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
            'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
            'j' => [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
            'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
            'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
            'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
            'n' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10001, 0b10001, 0b10001],
            'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
            'p' => [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
            'q' => [0b00000, 0b00000, 0b01101, 0b10011, 0b01111, 0b00001, 0b00001],
            'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
            's' => [0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110],
            't' => [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
            'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
            'v' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
            'w' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
            'x' => [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
            'y' => [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
            'z' => [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
            ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
            ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
            '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
            ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b01000],
            '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
            '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
            '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
            ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
            '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
            ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
            '{' => [0b00110, 0b00100, 0b00100, 0b01000, 0b00100, 0b00100, 0b00110],
            '}' => [0b01100, 0b00100, 0b00100, 0b00010, 0b00100, 0b00100, 0b01100],
            '/' => [0b00000, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000],
            '\\' => [0b00000, 0b10000, 0b01000, 0b00100, 0b00010, 0b00001, 0b00000],
            '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
            '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
            '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
            '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
            '?' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100],
            '@' => [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
            '#' => [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
            '$' => [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100],
            '%' => [0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011],
            '^' => [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
            '&' => [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101],
            '*' => [0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000],
            '~' => [0b00000, 0b00000, 0b01101, 0b10110, 0b00000, 0b00000, 0b00000],
            '`' => [0b01000, 0b00100, 0b00010, 0b00000, 0b00000, 0b00000, 0b00000],
            '\'' => [0b00100, 0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000],
            '"' => [0b01010, 0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000],
            _ => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        };

        // Draw the character pixel by pixel
        for row in 0..7 {
            for col in 0..5 {
                // Check if the pixel should be drawn (bit is set)
                if (font[row] >> (4 - col)) & 1 == 1 {
                    self.draw_pixel(frame, x + col as u32, y + row as u32, color);
                }
            }
        }
    }

    // Helper method to draw text
    fn draw_text(&self, frame: &mut [u8], x: u32, y: u32, text: &str, color: [u8; 4]) {
        let mut cursor_x = x;
        for c in text.chars() {
            self.draw_char(frame, cursor_x, y, c, color);
            cursor_x += 6; // 5 pixels for the character + 1 pixel spacing
        }
    }

    // Method to render status information in the top-left corner
    fn render_status(&self, frame: &mut [u8], color_cycling_enabled: bool) {
        // Background color for the status area (semi-transparent black)
        let bg_color = [0, 0, 0, 200];

        // Draw background rectangle for the status area
        let status_width = 300;
        let status_height = 130; // Increased height to accommodate the additional line
        for y in 0..status_height {
            for x in 0..status_width {
                self.draw_pixel(frame, x, y, bg_color);
            }
        }

        // Text color (white)
        let text_color = [255, 255, 255, 255];

        // Render each line of status information
        self.draw_text(frame, 10, 10, &format!("Position: ({:.3}, {:.3})", self.center_x, self.center_y), text_color);
        self.draw_text(frame, 10, 25, &format!("Zoom: {:.3}", self.zoom.log10()), text_color);
        self.draw_text(frame, 10, 40, &format!("Iterations: {}/{} (i: double, Shift+i: halve)", self.current_iterations, MAX_ITERATIONS), text_color);
        self.draw_text(frame, 10, 55, &format!("Color Cycling: {} (Toggle with 'C')", if color_cycling_enabled { "ON" } else { "OFF" }), text_color);
        self.draw_text(frame, 10, 70, &format!("Formula: {} (Toggle with 'F')", 
            if self.formula == 0 { "Z_n = Z_n-1^2 + C (standard)" } else { "Z_n = Z_n-1^2 - i*Z_n-1 + C (alternative)" }
        ), text_color);
        self.draw_text(frame, 10, 85, &format!("Progressive Rendering: {}", 
            if self.current_iterations < MAX_ITERATIONS { "Active (improving quality...)" } else { "Complete" }
        ), text_color);
        self.draw_text(frame, 10, 100, &format!("Orbit: {} points ({})", 
            self.orbit.len(),
            if self.orbit_converges { "converging - green" } else { "diverging - yellow" }
        ), text_color);
        self.draw_text(frame, 10, 115, &format!("Rendering: {} (Toggle with 'G')", 
            if self.use_gpu { "GPU (Compute Shader)" } else { "CPU (Parallel)" }
        ), text_color);
    }

    // Method to update the iteration count based on view changes
    fn update_iterations(&mut self) {
        // Check if view parameters have changed
        let view_changed = self.center_x != self.prev_center_x
            || self.center_y != self.prev_center_y
            || self.zoom != self.prev_zoom
            || self.formula != self.prev_formula;

        if view_changed {
             // Reset iterations to 1 if view has changed
             self.current_iterations = 3;
            //println!("View parameters changed - resetting iterations to 1");
            self.cache = vec![(0, 0.0, Complex64::new(0.0, 0.0)); (self.width * self.height) as usize]
         } else {
            //println!("View parameters unchanged - using current iterations: {}", self.current_iterations);
            // Progressively increase iterations
            if self.current_iterations < 256 {
                // Double iterations until 256
                self.current_iterations = std::cmp::min(
                    self.current_iterations.saturating_mul(2),
                    self.max_iterations);
            } else {
                // After 256, increase by 256 at a time, up to max_iterations
                self.current_iterations = std::cmp::min(
                    self.current_iterations.saturating_add(256),
                    self.max_iterations
                );
            }
        }

        // Update previous view parameters
        self.prev_center_x = self.center_x;
        self.prev_center_y = self.center_y;
        self.prev_zoom = self.zoom;
        self.prev_formula = self.formula;
    }

    fn render(&mut self, frame: &mut [u8], color_cycling_enabled: bool) {
        let start_time = Instant::now();

        if self.use_gpu {
            // GPU-based rendering
            if let Some(gpu) = &self.gpu_mandelbrot {
                // Update parameters
                let params = gpu_mandelbrot::MandelbrotParams {
                    center_x: self.center_x as f32,
                    center_y: self.center_y as f32,
                    zoom: self.zoom as f32,
                    max_iterations: self.current_iterations,
                    width: self.width,
                    height: self.height,
                    color_offset: self.color_offset as f32,
                    formula: self.formula as u32,
                };

                // Update GPU parameters
                gpu.update_params(params);

                // Run compute shader
                gpu.compute();

                // Copy results back to frame buffer
                gpu.copy_to_buffer(frame);

                let elapsed_time = start_time.elapsed();
                println!("GPU render call took: {:?}", elapsed_time);
            }
        } else {
            // CPU-based rendering (original implementation)
            // Create a vector to store pixel data in parallel
            let color_offset = self.color_offset;
            let current_iterations = self.current_iterations;
            let mut cache = std::mem::take(&mut self.cache);
            let width = self.width;

            let pixel_data: Vec<_> = cache.par_iter_mut()
                .enumerate()
                .map(|(i,(iterations, smooth_iter, z))| {
                    let x = (i % width as usize) as u32;
                    let y = (i / width as usize) as u32;
                    // Perform the calculation
                    let (new_iterations, new_smooth_iter, new_z) =
                        self.calculate(x, y, *iterations, current_iterations, *smooth_iter, *z);

                    // Update the cache values
                    *iterations = new_iterations;
                    *smooth_iter = new_smooth_iter;
                    *z = new_z;

                    if new_iterations == current_iterations {
                        // Points inside the Mandelbrot set
                        [0, 0, 0, 255]
                    } else {
                        // Points outside the set - use smooth coloring
                        let t = new_smooth_iter / current_iterations as f64;

                        // This creates a rainbow-like effect with smooth transitions
                        let hue = (0.95 * t * 360.0 + color_offset) % 360.0;
                        let (r, g, b) = Self::hsv_to_rgb(hue, 0.8, 1.0);

                        [r, g, b, 255]
                    }
                }).collect();

            self.cache = cache;

            let elapsed_time = start_time.elapsed();
            println!("CPU render call took: {:?}", elapsed_time);

            // Copy the calculated pixel data to the frame buffer
            for (i, pixel_color) in pixel_data.iter().enumerate() {
                let pixel_offset = i * 4;
                if pixel_offset + 3 < frame.len() {
                    frame[pixel_offset..pixel_offset + 4].copy_from_slice(pixel_color);
                }
            }
        }

        // Draw the orbit if available
        if !self.orbit.is_empty() {
            // Use different colors for converging and diverging orbits
            let orbit_color = if self.orbit_converges {
                [0, 255, 0, 255] // Green for converging orbits
            } else {
                [255, 255, 0, 255] // Yellow for diverging orbits
            };

            // Draw lines connecting the orbit points
            for i in 0..self.orbit.len() - 1 {
                let (real1, imag1) = self.orbit[i];
                let (real2, imag2) = self.orbit[i + 1];

                // Convert complex coordinates to screen coordinates
                let (x1, y1) = self.complex_to_screen(real1, imag1);
                let (x2, y2) = self.complex_to_screen(real2, imag2);

                // Draw a line connecting the points
                self.draw_line(frame, x1, y1, x2, y2, orbit_color);
            }

            // Draw points at each orbit position
            for &(real, imag) in &self.orbit {
                let (x, y) = self.complex_to_screen(real, imag);

                // Draw a small circle (3x3 pixels) at each point
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let px = (x as i32 + dx) as u32;
                        let py = (y as i32 + dy) as u32;
                        if px < self.width && py < self.height {
                            self.draw_pixel(frame, px, py, orbit_color);
                        }
                    }
                }
            }
        }

        // Render the status information in the top-left corner
        self.render_status(frame, color_cycling_enabled);

        // Update iterations based on view changes
        self.update_iterations();
    }

    // Helper function to convert HSV to RGB
    fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
        let c = v * s;
        let h_prime = h / 60.0;
        let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
        let m = v - c;

        let (r1, g1, b1) = match h_prime as u32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };

        let r = ((r1 + m) * 255.0) as u8;
        let g = ((g1 + m) * 255.0) as u8;
        let b = ((b1 + m) * 255.0) as u8;

        (r, g, b)
    }

    fn screen_to_complex(&self, screen_x: f64, screen_y: f64) -> (f64, f64) {
        let width = self.width as f64;
        let height = self.height as f64;
        let aspect_ratio = width / height;
        let scale_x = self.zoom * aspect_ratio;
        let scale_y = self.zoom;

        let real = self.center_x + (screen_x / width - 0.5) * scale_x;
        let imag = self.center_y + (screen_y / height - 0.5) * scale_y;

        (real, imag)
    }

    fn complex_to_screen(&self, real: f64, imag: f64) -> (u32, u32) {
        let width = self.width as f64;
        let height = self.height as f64;
        let aspect_ratio = width / height;
        let scale_x = self.zoom * aspect_ratio;
        let scale_y = self.zoom;

        let screen_x = ((real - self.center_x) / scale_x + 0.5) * width;
        let screen_y = ((imag - self.center_y) / scale_y + 0.5) * height;

        // Clamp to screen boundaries and convert to u32
        let x = screen_x.max(0.0).min(width - 1.0) as u32;
        let y = screen_y.max(0.0).min(height - 1.0) as u32;

        (x, y)
    }

    fn calculate_orbit(&self, real: f64, imag: f64) -> (Vec<(f64, f64)>, bool) {
        // Mandelbrot iteration using Complex numbers
        let c = Complex::new(real, imag);
        let mut z = Complex::new(0.0, 0.0);
        let mut orbit = Vec::with_capacity(self.current_iterations as usize);
        let mut converges = true;  // Assume the orbit converges until proven otherwise

        // Add the starting point (0,0)
        orbit.push((z.re, z.im));

        // Use current_iterations for consistency with calculate method
        for _ in 0..20 {
            // Iteration step using the helper function with Complex numbers
            z = self.iterate_point(z, c);

            // Add the new point to the orbit
            orbit.push((z.re, z.im));

            // Check if point escaped
            let mag_sq = z.norm_sqr();
            if mag_sq > ESCAPE_RADIUS_SQ {
                converges = false;  // The orbit diverges
                break;
            }
        }

        (orbit, converges)
    }

    fn zoom_in(&mut self, factor: f64, mouse_x: Option<f64>, mouse_y: Option<f64>) {
        if let (Some(x), Some(y)) = (mouse_x, mouse_y) {
            // Convert mouse position to complex coordinates before zooming
            let (mouse_complex_x, mouse_complex_y) = self.screen_to_complex(x, y);

            // Calculate vector from current center to mouse position
            let vector_x = mouse_complex_x - self.center_x;
            let vector_y = mouse_complex_y - self.center_y;

            // Adjust zoom level
            self.zoom *= factor;

            // Set new center - only move halfway toward the mouse position
            self.center_x += vector_x * 0.5;
            self.center_y += vector_y * 0.5;
        } else {
            // If no mouse position provided, just zoom around current center
            self.zoom *= factor;
        }
    }

    fn zoom_out(&mut self, factor: f64, mouse_x: Option<f64>, mouse_y: Option<f64>) {
        if let (Some(x), Some(y)) = (mouse_x, mouse_y) {
            // Convert mouse position to complex coordinates before zooming
            let (mouse_complex_x, mouse_complex_y) = self.screen_to_complex(x, y);

            // Calculate vector from current center to mouse position
            let vector_x = mouse_complex_x - self.center_x;
            let vector_y = mouse_complex_y - self.center_y;

            // Adjust zoom level
            self.zoom /= factor;

            // Set new center - only move halfway toward the mouse position
            self.center_x += vector_x * 0.5;
            self.center_y += vector_y * 0.5;
        } else {
            // If no mouse position provided, just zoom around current center
            self.zoom /= factor;
        }
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        // Scale the movement based on zoom level
        let aspect_ratio = self.width as f64 / self.height as f64;

        // Base pan factor - reduced to make panning less sensitive overall
        let base_pan_factor = 1.0/self.height as f64;

        // Calculate adjusted pan factor - reduce sensitivity when zoomed out
        // The initial zoom is 4.0, so we use that as a reference point
        let initial_zoom = 4.0;

        // Invert the relationship: when zoom is high (zoomed out), sensitivity should be low
        // When zoom is low (zoomed in), sensitivity should be higher
        let zoom_ratio = if self.zoom > initial_zoom {
            // When zoomed out (zoom > initial_zoom), reduce sensitivity
            initial_zoom / self.zoom
        } else {
            // When zoomed in (zoom < initial_zoom), keep normal or slightly increased sensitivity
            1.0
        };

        // Apply the pan factor, scaled by zoom ratio
        let adjusted_pan_factor = base_pan_factor * zoom_ratio;

        self.center_x += dx * self.zoom * aspect_ratio * adjusted_pan_factor;
        self.center_y += dy * self.zoom * adjusted_pan_factor;
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        // Only resize if dimensions have actually changed
        if self.width == new_width && self.height == new_height {
            return;
        }

        // Update dimensions
        self.width = new_width;
        self.height = new_height;

        // Resize the cache for CPU rendering
        self.cache = vec![(0, 0.0, Complex64::new(0.0, 0.0)); (new_width * new_height) as usize];

        // Resize GPU buffers if using GPU
        if let Some(gpu) = &mut self.gpu_mandelbrot {
            gpu.resize(new_width, new_height);
        }

        // Force a recalculation of the view by setting previous parameters to different values
        self.prev_center_x = self.center_x + 1.0;
        self.prev_center_y = self.center_y + 1.0;
        self.prev_zoom = self.zoom * 2.0;
    }

    // Toggle between CPU and GPU rendering
    fn toggle_gpu(&mut self) {
        self.use_gpu = !self.use_gpu;
        println!("Using {} rendering", if self.use_gpu { "GPU" } else { "CPU" });
    }
}

fn main() -> Result<(), Error> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Apfelmännchen (Mandelbrot Set)")
            .with_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(window_size.width, window_size.height, surface_texture)?
    };

    let window_size = window.inner_size();
    let mut mandelbrot = Mandelbrot::new();
    mandelbrot.resize(window_size.width, window_size.height);
    let mut color_cycling_enabled = true;

    // Variables for tracking mouse panning
    let mut is_panning = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;

    event_loop.run(move |event, _, control_flow| {
        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() || input.destroyed() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Toggle color cycling with 'C' key
            if input.key_pressed(VirtualKeyCode::C) {
                color_cycling_enabled = !color_cycling_enabled;
            }

            // Toggle formula with 'F' key
            if input.key_pressed(VirtualKeyCode::F) {
                mandelbrot.formula = 1 - mandelbrot.formula; // Toggle between 0 and 1
            }

            // Toggle between CPU and GPU rendering with 'G' key
            if input.key_pressed(VirtualKeyCode::G) {
                mandelbrot.toggle_gpu();
            }

            // Double max_iterations with 'i' key
            if input.key_pressed(VirtualKeyCode::I) && !input.key_held(VirtualKeyCode::LShift) && !input.key_held(VirtualKeyCode::RShift) {
                mandelbrot.max_iterations *= 2;
            }

            // Halve max_iterations with 'Shift+i' key
            if input.key_pressed(VirtualKeyCode::I) && (input.key_held(VirtualKeyCode::LShift) || input.key_held(VirtualKeyCode::RShift)) {
                // Ensure we don't go below 1
                mandelbrot.max_iterations = std::cmp::max(1, mandelbrot.max_iterations / 2);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                // Resize the pixel buffer to match the new window size
                if let Err(err) = pixels.resize_buffer(size.width, size.height) {
                    error!("pixels.resize_buffer error: {err}");
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                // Resize the surface texture to match the new window size
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    error!("pixels.resize_surface error: {err}");
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                // Update the Mandelbrot struct with the new window dimensions
                mandelbrot.resize(size.width, size.height);
            }

            // Handle zooming
            if input.scroll_diff().abs() > 0.0 {
                let zoom_factor = 0.8;
                // Get current mouse position
                let mouse_pos = input.mouse();

                // Convert f32 coordinates to f64
                let mouse_x = mouse_pos.map(|(x, _)| x as f64);
                let mouse_y = mouse_pos.map(|(_, y)| y as f64);

                if input.scroll_diff() > 0.0 {
                    mandelbrot.zoom_in(zoom_factor, mouse_x, mouse_y);
                } else {
                    mandelbrot.zoom_out(zoom_factor, mouse_x, mouse_y);
                }
            }

            // Handle panning with arrow keys
            let mut dx = 0.0;
            let mut dy = 0.0;

            if input.key_held(VirtualKeyCode::Left) {
                dx -= 1.0;
            }
            if input.key_held(VirtualKeyCode::Right) {
                dx += 1.0;
            }
            if input.key_held(VirtualKeyCode::Up) {
                dy -= 1.0;
            }
            if input.key_held(VirtualKeyCode::Down) {
                dy += 1.0;
            }

            // Handle panning with left mouse button
            let current_mouse_pos = input.mouse().map(|(x, y)| (x as f64, y as f64));

            // Update the mouse position in the Mandelbrot struct
            mandelbrot.mouse_pos = current_mouse_pos;

            // Calculate the orbit for the point under the mouse cursor
            if let Some((x, y)) = current_mouse_pos {
                let (real, imag) = mandelbrot.screen_to_complex(x, y);
                let (orbit, converges) = mandelbrot.calculate_orbit(real, imag);
                mandelbrot.orbit = orbit;
                mandelbrot.orbit_converges = converges;
            }

            // Check for left mouse button press/release
            if input.mouse_pressed(0) { // 0 is the left mouse button
                is_panning = true;
                last_mouse_pos = current_mouse_pos;
            } else if input.mouse_released(0) {
                is_panning = false;
                last_mouse_pos = None;
            }

            // If panning (left mouse button held) and we have a previous position
            if is_panning && input.mouse_held(0) {
                if let (Some(current_pos), Some(last_pos)) = (current_mouse_pos, last_mouse_pos) {
                    // Calculate the movement delta
                    let mouse_dx = last_pos.0 - current_pos.0;
                    let mouse_dy = last_pos.1 - current_pos.1;

                    // Add to the existing keyboard movement
                    dx += mouse_dx;
                    dy += mouse_dy;

                    // Update the last position
                    last_mouse_pos = current_mouse_pos;
                }
            }

            if dx != 0.0 || dy != 0.0 {
                mandelbrot.pan(dx, dy);
            }

            // Update the window title to a simpler version (detailed info is now rendered on the image)
            window.set_title("Apfelmännchen (Mandelbrot Set)");

            // Update color cycling if enabled
            if color_cycling_enabled {
                mandelbrot.cycle_colors(0.5); // Adjust this value to control cycling speed
            }

            // Render the current frame
            mandelbrot.render(pixels.frame_mut(), color_cycling_enabled);

            // Render the frame
            if let Err(err) = pixels.render() {
                error!("pixels.render error: {err}");
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Set the control flow to wait for the next event
        // Update iterations based on view changes
        *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100));
    });
}
