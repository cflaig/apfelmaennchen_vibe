use std::borrow::Cow;
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, Queue, ShaderModule,
};
use bytemuck::{Pod, Zeroable};

// Parameters for the Mandelbrot calculation
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MandelbrotParams {
    pub center_x: f32,
    pub center_y: f32,
    pub zoom: f32,
    pub max_iterations: u32,
    pub width: u32,
    pub height: u32,
    pub color_offset: f32,
    pub formula: u32,
}

pub struct GpuMandelbrot {
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
    params_buffer: Buffer,
    output_buffer: Buffer,
    bind_group: BindGroup,
    width: u32,
    height: u32,
}

impl GpuMandelbrot {
    pub async fn new(width: u32, height: u32) -> Self {
        // Create instance
        let instance = wgpu::Instance::default();

        // Create adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(), // No special features requested
                    limits: wgpu::Limits::default(),  // Use default resource limits
                },
                None,
            )
            .await
            .expect("Failed to create device");


        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mandelbrot Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("mandelbrot.wgsl"))),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mandelbrot Bind Group Layout"),
            entries: &[
                // Params buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mandelbrot Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mandelbrot Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Create params buffer with initial values
        let params = MandelbrotParams {
            center_x: -0.5,
            center_y: 0.0,
            zoom: 4.0,
            max_iterations: 20,
            width,
            height,
            color_offset: 0.0,
            formula: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mandelbrot Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create output buffer
        let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mandelbrot Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for reading back the results
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mandelbrot Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mandelbrot Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            compute_pipeline,
            params_buffer,
            output_buffer,
            bind_group,
            width,
            height,
        }
    }

    pub fn update_params(&self, params: MandelbrotParams) {
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    pub fn compute(&self) {
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mandelbrot Compute Encoder"),
        });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mandelbrot Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            // Dispatch workgroups
            let workgroup_size = 16;
            let workgroup_count_x = (self.width + workgroup_size - 1) / workgroup_size;
            let workgroup_count_y = (self.height + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn copy_to_buffer(&self, buffer: &mut [u8]) {
        // Create a staging buffer for reading back the results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mandelbrot Staging Buffer"),
            size: (self.width * self.height * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mandelbrot Copy Encoder"),
        });

        // Copy from output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            (self.width * self.height * 4) as wgpu::BufferAddress,
        );

        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer to read the results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        // Wait for the mapping to complete
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(async {
            receiver.receive().await.unwrap().unwrap();
        });

        // Read the data from the staging buffer
        let data = buffer_slice.get_mapped_range();
        buffer.copy_from_slice(&data);

        // Unmap the staging buffer
        drop(data);
        staging_buffer.unmap();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Recreate output buffer with new size
        let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
        self.output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mandelbrot Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Recreate bind group with new output buffer
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mandelbrot Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });
    }
}