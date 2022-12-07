#include <metal_stdlib>
using namespace metal;
using namespace simd;

constant uint UINT16_MAX_VAL = 65535;
constant float PI = 3.14159265358979323846;


// ===========================================================================================================
// Helper functions
// ===========================================================================================================

kernel void fill_with_zeros(texture2d<float, access::write> texture [[texture(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    texture.write(0, gid);
}


kernel void texture_uint16_to_float(texture2d<uint, access::read> in_texture [[texture(0)]],
                                    texture2d<float, access::write> out_texture [[texture(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    uint4 val_unint = in_texture.read(gid);
    float4 val_float = float4(val_unint); // removed scaling to the range 0 to 1
    out_texture.write(val_float, gid); 
}


kernel void convert_uint16(texture2d<float, access::read> in_texture [[texture(0)]],
                           texture2d<uint, access::write> out_texture [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
      
    float color_value = in_texture.read(gid).r;
        
    color_value = clamp(color_value, 0.0f, float(UINT16_MAX_VAL));
    
    uint out = int(round(color_value));
    
    out_texture.write(out, gid);
}


kernel void upsample_nearest_int(texture2d<int, access::read> in_texture [[texture(0)]],
                                 texture2d<int, access::write> out_texture [[texture(1)]],
                                 constant float& scale_x [[buffer(0)]],
                                 constant float& scale_y [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]]) {

    int x = int(round(float(gid.x) / scale_x));
    int y = int(round(float(gid.y) / scale_y));
    int4 out_color = in_texture.read(uint2(x, y));
    out_texture.write(out_color, gid);
}


kernel void upsample_bilinear_float(texture2d<float, access::read> in_texture [[texture(0)]],
                                    texture2d<float, access::write> out_texture [[texture(1)]],
                                    constant float& scale_x [[buffer(0)]],
                                    constant float& scale_y [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    // naming based on https://en.wikipedia.org/wiki/Bilinear_interpolation#/media/File:BilinearInterpolation.svg
    
    float x = float(gid.x) / scale_x;
    float y = float(gid.y) / scale_y;
    float epsilon = 1e-5;
    
    // interpolate over the x-axis
    float4 i1, i2;
    if (abs(x - round(x)) < epsilon) {
        i1 = float4(in_texture.read(uint2(round(x), floor(y))));
        i2 = float4(in_texture.read(uint2(round(x), ceil(y) )));
    } else {
        float4 i11 = float4(in_texture.read(uint2(floor(x), floor(y))));
        float4 i12 = float4(in_texture.read(uint2(floor(x), ceil(y) )));
        float4 i21 = float4(in_texture.read(uint2(ceil(x),  floor(y))));
        float4 i22 = float4(in_texture.read(uint2(ceil(x),  ceil(y) )));
        i1 = (ceil(x) - x) * i11 + (x - floor(x)) * i21;
        i2 = (ceil(x) - x) * i12 + (x - floor(x)) * i22;
    }
    
    // interpolate over the y-axis
    float4 i;
    if (abs(y - round(y)) < epsilon) {
        i = i1;
    } else {
        i = (ceil(y) - y) * i1 + (y - floor(y)) * i2;
    }
    
    out_texture.write(i, gid);
}


kernel void avg_pool(texture2d<float, access::read> in_texture [[texture(0)]],
                     texture2d<float, access::write> out_texture [[texture(1)]],
                     constant int& scale [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float out_pixel = 0;
    int x0 = gid.x * scale;
    int y0 = gid.y * scale;
    for (int dx = 0; dx < scale; dx++) {
        for (int dy = 0; dy < scale; dy++) {
            int x = x0 + dx;
            int y = y0 + dy;
            out_pixel += in_texture.read(uint2(x, y)).r;
        }
    }
    out_pixel /= (scale*scale);
    out_texture.write(out_pixel, gid);
}


kernel void blur_mosaic_x(texture2d<float, access::read> in_texture [[texture(0)]],
                          texture2d<float, access::write> out_texture [[texture(1)]],
                          constant int& kernel_size [[buffer(0)]],
                          constant int& mosaic_pettern_width [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    int texture_width = in_texture.get_width();
    
    // set kernel weights of binomial filter and clamp kernel_size to a maximum of 8
    int const kernel_size_clamped = clamp(kernel_size, 0, 8);
   
    float3x3 binomial_weights = float3x3({1, 0, 0}, {0, 0, 0}, {0, 0, 0});

    if(kernel_size_clamped == 1)
        binomial_weights = float3x3({2, 1, 0}, {0, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 2)
        binomial_weights = float3x3({6, 4, 1}, {0, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 3)
        binomial_weights = float3x3({20, 15, 6}, {1, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 4)
        binomial_weights = float3x3({70, 56, 28}, {8, 1, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 5)
        binomial_weights = float3x3({252, 210, 120}, {45, 10, 1}, {0, 0, 0});
    else if(kernel_size_clamped == 6)
        binomial_weights = float3x3({924, 792, 495}, {220, 66, 12}, {1, 0, 0});
    else if(kernel_size_clamped == 7)
        binomial_weights = float3x3({3432, 3003, 2002}, {1001, 364, 91}, {14, 1, 0});
    else if(kernel_size_clamped == 8)
        binomial_weights = float3x3({12870, 11440, 8008}, {4368, 1820, 560}, {120, 16, 1});
        
    // compute a sigle output pixel
    float total_intensity = 0;
    float total_weight = 0;
    int y = gid.y;
    for (int dx = -kernel_size_clamped; dx <= kernel_size_clamped; dx++) {
        int x = gid.x + mosaic_pettern_width*dx;
        if (0 <= x && x < texture_width) {
            
            int const m = int(abs(dx)/3.0f+0.1f);
            int const n = int(abs(dx) % 3);
            
            float const weight = float(binomial_weights[m][n]);
            total_intensity += weight * in_texture.read(uint2(x, y)).r;
            total_weight += weight;
        }
    }
    
    // write output pixel
    float out_intensity = total_intensity / total_weight;
    out_texture.write(out_intensity, gid);
}


kernel void blur_mosaic_y(texture2d<float, access::read> in_texture [[texture(0)]],
                          texture2d<float, access::write> out_texture [[texture(1)]],
                          constant int& kernel_size [[buffer(0)]],
                          constant int& mosaic_pettern_width [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    int texture_height = in_texture.get_height();
    
    // set kernel weights of binomial filter and clamp kernel_size to a maximum of 8
    int const kernel_size_clamped = clamp(kernel_size, 0, 8);
   
    float3x3 binomial_weights = float3x3({1, 0, 0}, {0, 0, 0}, {0, 0, 0});

    if(kernel_size_clamped == 1)
        binomial_weights = float3x3({2, 1, 0}, {0, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 2)
        binomial_weights = float3x3({6, 4, 1}, {0, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 3)
        binomial_weights = float3x3({20, 15, 6}, {1, 0, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 4)
        binomial_weights = float3x3({70, 56, 28}, {8, 1, 0}, {0, 0, 0});
    else if(kernel_size_clamped == 5)
        binomial_weights = float3x3({252, 210, 120}, {45, 10, 1}, {0, 0, 0});
    else if(kernel_size_clamped == 6)
        binomial_weights = float3x3({924, 792, 495}, {220, 66, 12}, {1, 0, 0});
    else if(kernel_size_clamped == 7)
        binomial_weights = float3x3({3432, 3003, 2002}, {1001, 364, 91}, {14, 1, 0});
    else if(kernel_size_clamped == 8)
        binomial_weights = float3x3({12870, 11440, 8008}, {4368, 1820, 560}, {120, 16, 1});
        
    // compute a sigle output pixel
    float total_intensity = 0;
    float total_weight = 0;
    int x = gid.x;
    for (int dy = -kernel_size_clamped; dy <= kernel_size_clamped; dy++) {
        int y = gid.y + mosaic_pettern_width*dy;
        if (0 <= y && y < texture_height) {
           
            int const m = int(abs(dy)/3.0f+0.1f);
            int const n = int(abs(dy) % 3);
            
            float const weight = float(binomial_weights[m][n]);
            total_intensity += weight * in_texture.read(uint2(x, y)).r;
            total_weight += weight;
        }
    }
    
    // write output pixel
    float out_intensity = total_intensity / total_weight;
    out_texture.write(out_intensity, gid);
}


kernel void extend_texture(texture2d<float, access::read> in_texture [[texture(0)]],
                           texture2d<float, access::write> out_texture [[texture(1)]],
                           constant int& pad_left [[buffer(0)]],
                           constant int& pad_top [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
        
    int x = gid.x + pad_left;
    int y = gid.y + pad_top;
 
    float color_value = in_texture.read(gid).r;
    
    out_texture.write(color_value, uint2(x, y));
}


kernel void crop_texture(texture2d<float, access::read> in_texture [[texture(0)]],
                         texture2d<float, access::write> out_texture [[texture(1)]],
                         constant int& pad_left [[buffer(0)]],
                         constant int& pad_top [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
      
    int x = gid.x + pad_left;
    int y = gid.y + pad_top;
  
    float color_value = in_texture.read(uint2(x, y)).r;
    
    out_texture.write(color_value, gid);
}




kernel void add_texture(texture2d<float, access::read> in_texture [[texture(0)]],
                        texture2d<float, access::read_write> out_texture [[texture(1)]],
                        constant int& n_textures [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    float color_value = out_texture.read(gid).r + in_texture.read(gid).r/float(n_textures);
    
    out_texture.write(color_value, gid);
}


kernel void copy_texture(texture2d<float, access::read> in_texture [[texture(0)]],
                         texture2d<float, access::write> out_texture [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {

    out_texture.write(in_texture.read(gid), gid);
}


kernel void convert_rgba(texture2d<float, access::read> in_texture [[texture(0)]],
                         texture2d<float, access::write> out_texture [[texture(1)]],
                         constant int& crop_merge_x [[buffer(0)]],
                         constant int& crop_merge_y [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    int const m = gid.x*2 + crop_merge_x;
    int const n = gid.y*2 + crop_merge_y;
    
    float4 color_value = float4(in_texture.read(uint2(m, n)).r,   in_texture.read(uint2(m+1, n)).r,
                                in_texture.read(uint2(m, n+1)).r, in_texture.read(uint2(m+1, n+1)).r);
    
    out_texture.write(color_value, gid);
}


kernel void convert_bayer(texture2d<float, access::read> in_texture [[texture(0)]],
                          texture2d<float, access::write> out_texture [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    int const m = gid.x*2;
    int const n = gid.y*2;
    
    float4 color_value = float4(in_texture.read(gid));
     
    out_texture.write(color_value[0], uint2(m,   n));
    out_texture.write(color_value[1], uint2(m+1, n));
    out_texture.write(color_value[2], uint2(m,   n+1));
    out_texture.write(color_value[3], uint2(m+1, n+1));
}


// ===========================================================================================================
// Functions specific to image alignment
// ===========================================================================================================

kernel void compute_tile_differences(texture2d<float, access::read> ref_texture [[texture(0)]],
                                     texture2d<float, access::read> comp_texture [[texture(1)]],
                                     texture2d<int, access::read> prev_alignment [[texture(2)]],
                                     texture3d<float, access::write> tile_diff [[texture(3)]],
                                     constant int& downscale_factor [[buffer(0)]],
                                     constant int& tile_size [[buffer(1)]],
                                     constant int& search_dist [[buffer(2)]],
                                     constant int& n_tiles_x [[buffer(3)]],
                                     constant int& n_tiles_y [[buffer(4)]],
                                     uint3 gid [[thread_position_in_grid]]) {
    
    // load args
    int texture_width = ref_texture.get_width();
    int texture_height = ref_texture.get_height();
    int tile_half_size = tile_size / 2;
    int n_pos_1d = 2*search_dist + 1;
    
    // compute tile position if previous alignment were 0
    int x0 = int(floor( tile_half_size + float(gid.x)/float(n_tiles_x-1) * (texture_width  - tile_size - 1) ));
    int y0 = int(floor( tile_half_size + float(gid.y)/float(n_tiles_y-1) * (texture_height - tile_size - 1) ));
    
    // compute current tile displacement based on thread index
    int dy0 = gid.z / n_pos_1d - search_dist;
    int dx0 = gid.z % n_pos_1d - search_dist;
    
    // factor in previous alignmnet
    int4 prev_align = prev_alignment.read(uint2(gid.x, gid.y));
    dx0 += downscale_factor * prev_align.x;
    dy0 += downscale_factor * prev_align.y;
    
    // compute tile difference
    float diff = 0;
    for (int dx1 = -tile_half_size; dx1 < tile_half_size; dx1++){
        for (int dy1 = -tile_half_size; dy1 < tile_half_size; dy1++){
            // compute the indices of the pixels to compare
            int ref_tile_x = x0 + dx1;
            int ref_tile_y = y0 + dy1;
            int comp_tile_x = ref_tile_x + dx0;
            int comp_tile_y = ref_tile_y + dy0;
            
            // if the comparison pixels are outside of the frame, attach a high loss to them
            if ((comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height)) {
                diff += (2*UINT16_MAX_VAL); // value has to be increased as scaling of values to the range of 0 to 1 was removed
            } else {
                diff += abs(ref_texture.read(uint2(ref_tile_x, ref_tile_y)).r - comp_texture.read(uint2(comp_tile_x, comp_tile_y)).r);
            }
        }
    }
    
    // store tile difference
    tile_diff.write(diff, gid);
}


kernel void compute_tile_alignments(texture3d<float, access::read> tile_diff [[texture(0)]],
                                    texture2d<int, access::read> prev_alignment [[texture(1)]],
                                    texture2d<int, access::write> current_alignment [[texture(2)]],
                                    constant int& downscale_factor [[buffer(0)]],
                                    constant int& search_dist [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    // load args
    int n_pos_1d = 2*search_dist + 1;
    int n_pos_2d = n_pos_1d * n_pos_1d;
    
    // find tile position with the lowest pixel difference
    float current_diff;
    float min_diff_val = tile_diff.read(uint3(gid.x, gid.y, 0)).r;
    int min_diff_idx = 0;
    for (int i = 0; i < n_pos_2d; i++) {
        current_diff = tile_diff.read(uint3(gid.x, gid.y, i)).r;
        if (current_diff < min_diff_val) {
            min_diff_val = current_diff;
            min_diff_idx = i;
        }
    }
    
    // compute tile displacement if previous alignment were 0
    int dy = min_diff_idx / n_pos_1d - search_dist;
    int dx = min_diff_idx % n_pos_1d - search_dist;
    
    // factor in previous alignmnet
    int4 prev_align = prev_alignment.read(gid);
    dx += downscale_factor * prev_align.x;
    dy += downscale_factor * prev_align.y;
    
    // store alignment
    int4 out = int4(dx, dy, 0, 0);
    current_alignment.write(out, gid);
}


kernel void correct_upsampling_error(texture2d<float, access::read> ref_texture [[texture(0)]],
                                     texture2d<float, access::read> comp_texture [[texture(1)]],
                                     texture2d<int, access::read> prev_alignment [[texture(2)]],
                                     texture2d<int, access::write> prev_alignment_corrected [[texture(3)]],
                                     constant int& downscale_factor [[buffer(0)]],
                                     constant int& tile_size [[buffer(1)]],
                                     constant int& n_tiles_x [[buffer(2)]],
                                     constant int& n_tiles_y [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    int texture_width = ref_texture.get_width();
    int texture_height = ref_texture.get_height();
    int tile_half_size = tile_size / 2;
    
    // compute tile position if previous alignment were 0
    int x0 = int(floor( tile_half_size + float(gid.x)/float(n_tiles_x-1) * (texture_width  - tile_size - 1) ));
    int y0 = int(floor( tile_half_size + float(gid.y)/float(n_tiles_y-1) * (texture_height - tile_size - 1) ));

    // calculate shifts for 3 candidate alignments to evaluate
    int3 const x_shift = int3(0, ((gid.x%2 == 0) ? -1 : 1), 0);
    int3 const y_shift = int3(0, 0, ((gid.y%2 == 0) ? -1 : 1));
    
    float best_diff = float(1e20);
    int4 best_align = int4(0, 0, 0, 0);
    
    int3 const x = clamp(int3(gid.x+x_shift), 0, n_tiles_x-1);
    int3 const y = clamp(int3(gid.y+y_shift), 0, n_tiles_y-1);
    
    for (int i=0; i < 3; i++)
    {
        // factor in previous alignmnet
        int4 prev_align = prev_alignment.read(uint2(x[i], y[i]));
        int dx0 = downscale_factor * prev_align.x;
        int dy0 = downscale_factor * prev_align.y;
        
        // compute tile difference
        float diff = 0;
        for (int dx1 = -tile_half_size; dx1 < tile_half_size; dx1++){
            for (int dy1 = -tile_half_size; dy1 < tile_half_size; dy1++){
                // compute the indices of the pixels to compare
                int ref_tile_x = x0 + dx1;
                int ref_tile_y = y0 + dy1;
                int comp_tile_x = ref_tile_x + dx0;
                int comp_tile_y = ref_tile_y + dy0;
                
                // if the comparison pixels are outside of the frame, attach a high loss to them
                if ((comp_tile_x < 0) || (comp_tile_y < 0) || (comp_tile_x >= texture_width) || (comp_tile_y >= texture_height)) {
                    diff += (2*UINT16_MAX_VAL); // value has to be increased as scaling of values to the range of 0 to 1 was removed
                } else {
                    diff += abs(ref_texture.read(uint2(ref_tile_x, ref_tile_y)).r - comp_texture.read(uint2(comp_tile_x, comp_tile_y)).r);
                }
            }
        }
        
        if(diff < best_diff)
        {
            best_diff = diff;
            best_align = prev_align;
        }
    }
    
    // store corrected alignment
    prev_alignment_corrected.write(best_align, gid);
}


kernel void warp_texture(texture2d<float, access::read> in_texture [[texture(0)]],
                         texture2d<float, access::read_write> out_texture [[texture(1)]],
                         texture2d<int, access::read> prev_alignment [[texture(2)]],
                         constant int& downscale_factor [[buffer(0)]],
                         constant int& tile_size [[buffer(1)]],
                         constant int& n_tiles_x [[buffer(2)]],
                         constant int& n_tiles_y [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    int texture_width = in_texture.get_width();
    int texture_height = in_texture.get_height();
    int tile_half_size = tile_size / 2;
    
    // load coordinates of output pixel
    int x1_pix = gid.x;
    int y1_pix = gid.y;
    
    // compute the coordinates of output pixel in tile-grid units
    float x1_grid = float(x1_pix - tile_half_size) / float(texture_width  - tile_size - 1) * (n_tiles_x - 1);
    float y1_grid = float(y1_pix - tile_half_size) / float(texture_height - tile_size - 1) * (n_tiles_y - 1);
    
    // comput the two possible tile-grid indices that the given output pixel belongs to
    int x_grid_list[] = {int(floor(x1_grid)), int(floor(x1_grid)), int(ceil (x1_grid)), int(ceil(x1_grid))};
    int y_grid_list[] = {int(floor(y1_grid)), int(ceil (y1_grid)), int(floor(y1_grid)), int(ceil(y1_grid))};
    
    // loop over the two possible tile-grid indices that the given output pixel belongs to
    float total_intensity = 0;
    float total_weight = 0;
    for (int i = 0; i < 4; i++){
        
        // load the index of the tile
        int x_grid = x_grid_list[i];
        int y_grid = y_grid_list[i];
        
        // compute the pixel coordinates of the center of the reference tile
        int x0_pix = int(floor( tile_half_size + float(x_grid)/float(n_tiles_x-1) * (texture_width  - tile_size - 1) ));
        int y0_pix = int(floor( tile_half_size + float(y_grid)/float(n_tiles_y-1) * (texture_height - tile_size - 1) ));
        
        // check that the output pixel falls within the reference tile
        if ((abs(x1_pix - x0_pix) <= tile_half_size) && (abs(y1_pix - y0_pix) <= tile_half_size)) {
            
            // compute tile displacement
            int4 prev_align = prev_alignment.read(uint2(x_grid, y_grid));
            int dx = downscale_factor * prev_align.x;
            int dy = downscale_factor * prev_align.y;

            // load coordinates of the corresponding pixel from the comparison tile
            int x2_pix = x1_pix + dx;
            int y2_pix = y1_pix + dy;
            
            // compute the weight of the aligned pixel (based on distance from tile center)
            int dist_x = abs(x1_pix - x0_pix);
            int dist_y = abs(y1_pix - y0_pix);
            float weight_x = tile_size - dist_x - dist_y;
            float weight_y = tile_size - dist_x - dist_y;
            float curr_weight = weight_x * weight_y;
            total_weight += curr_weight;
            
            // add pixel value to the output
            total_intensity += curr_weight * in_texture.read(uint2(x2_pix, y2_pix)).r;
        }
    }
    
    // write output pixel
    float out_intensity = total_intensity / total_weight;
    out_texture.write(out_intensity, uint2(x1_pix, y1_pix));
}


// ===========================================================================================================
// Function specific to merging in the spatial domain
// ===========================================================================================================

kernel void add_textures_weighted(texture2d<float, access::read> texture1 [[texture(0)]],
                                  texture2d<float, access::read> texture2 [[texture(1)]],
                                  texture2d<float, access::read> weight_texture [[texture(2)]],
                                  texture2d<float, access::write> out_texture [[texture(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    float intensity1 = texture1.read(gid).r;
    float intensity2 = texture2.read(gid).r;
    float weight = weight_texture.read(gid).r;
    float out_intensity = weight * intensity2 + (1 - weight) * intensity1;
    out_texture.write(out_intensity, gid);
}


kernel void average_x(texture1d<float, access::read> in_texture [[texture(0)]],
                      device float *out_buffer [[buffer(0)]],
                      constant int& width [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    float total = 0;
    for (int x = 0; x < width; x++) {
        total += in_texture.read(uint(x)).r;
    }
    float avg = total / width;
    out_buffer[0] = avg;
}


kernel void average_y(texture2d<float, access::read> in_texture [[texture(0)]],
                      texture1d<float, access::write> out_texture [[texture(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x = gid;
    int texture_height = in_texture.get_height();
    float total = 0;
    for (int y = 0; y < texture_height; y++) {
        total += in_texture.read(uint2(x, y)).r;
    }
    float avg = total / texture_height;
    out_texture.write(avg, x);
}


kernel void color_difference(texture2d<float, access::read> texture1 [[texture(0)]],
                             texture2d<float, access::read> texture2 [[texture(1)]],
                             texture2d<float, access::write> out_texture [[texture(2)]],
                             constant int& mosaic_pettern_width [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
    float total_diff = 0;
    int x0 = gid.x * mosaic_pettern_width;
    int y0 = gid.y * mosaic_pettern_width;
    for (int dx = 0; dx < mosaic_pettern_width; dx++) {
        for (int dy = 0; dy < mosaic_pettern_width; dy++) {
            int x = x0 + dx;
            int y = y0 + dy;
            float i1 = texture1.read(uint2(x, y)).r;
            float i2 = texture2.read(uint2(x, y)).r;
            total_diff += abs(i1 - i2);
        }
    }
    out_texture.write(total_diff, gid);
}


kernel void compute_merge_weight(texture2d<float, access::read> texture_diff [[texture(0)]],
                                 texture2d<float, access::write> weight_texture [[texture(1)]],
                                 constant float* noise_sd_buffer [[buffer(0)]],
                                 constant float& robustness [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    float noise_sd = noise_sd_buffer[0];
    
    // load texture difference
    float diff = texture_diff.read(gid).r;
    
    // compute the weight to assign to the comparison frame
    // weight == 0 means that the aligned image is ignored
    // weight == 1 means that the aligned image has full weight
    float weight;
    if (robustness == 0) {
        // robustness == 0 means that robust merge is turned off
        weight = 1;
    } else {
        // compare the difference to image noise
        // as diff increases, the weight of the aligned image will continuously decrease from 1.0 to 0.0
        // the two extreme cases are:
        // diff == 0                   --> aligned image will have weight 1.0
        // diff >= noise_sd/robustness --> aligned image will have weight 0.0
        float max_diff = noise_sd / robustness;
        weight =  1 - diff / max_diff;
        weight = clamp(weight, 0.0, 1.0);
    }
    
    // write weight
    weight_texture.write(weight, gid);
}


// ===========================================================================================================
// Function specific to merging in the frequency domain
// ===========================================================================================================

kernel void calculate_rms_rgba(texture2d<float, access::read> ref_texture [[texture(0)]],
                               texture2d<float, access::write> rms_texture [[texture(1)]],
                               constant int& tile_size [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    
    // compute tile positions from gid
    int const m0 = gid.x*tile_size;
    int const n0 = gid.y*tile_size;
    
    // fill with zeros
    float4 noise_est = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // use 16x16 tile size here
    for (int dy = 0; dy < tile_size; dy++) {
        for (int dx = 0; dx < tile_size; dx++) {
            
            int const x = m0 + dx;
            int const y = n0 + dy;
            
            float4 const data_noise = ref_texture.read(uint2(x, y));
            
            noise_est += (data_noise * data_noise);
        }
    }

    noise_est = 0.25f*sqrt(noise_est)/float(tile_size);
    
    rms_texture.write(noise_est, gid);
}
    

kernel void forward_dft(texture2d<float, access::read> in_texture [[texture(0)]],
                        texture2d<float, access::read_write> tmp_texture_ft [[texture(1)]],
                        texture2d<float, access::write> out_texture_ft [[texture(2)]],
                        constant int& mosaic_pettern_width [[buffer(0)]],
                        constant int& tile_size [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    
    // compute tile positions from gid
    int const m0 = gid.x*tile_size;
    int const n0 = gid.y*tile_size;
    
    // pre-calculate factors for sine and cosine calculation
    float const angle = -2*PI/float(tile_size);
    
    // pre-initalize some vectors
    float4 const zeros = float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (int dm = 0; dm < tile_size; dm++) {
        // exploit symmetry of real dft and calculate reduced number of rows
        for (int dn = 0; dn <= tile_size/2; dn++) {
             
            int const m = m0 + dm;
            int const n = n0 + dn;
            
            // fill with zeros
            float4 Re = zeros;
            float4 Im = zeros;
            
            for (int dy = 0; dy < tile_size; dy++) {
                                  
                int const y = n0 + dy;
                
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts
                float const norm_cosine = (0.5f-0.500f*cos(-angle*(dm+0.5f)))*(0.5f-0.500f*cos(-angle*(dy+0.5f)));
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts (slightly adapted compared to original publication)
                //float const norm_cosine = (0.5f-0.505f*cos(-angle*(dm+0.5f)))*(0.5f-0.505f*cos(-angle*(dy+0.5f)));
                
                // calculate coefficients
                float const coefRe = cos(angle*dn*dy);
                float const coefIm = sin(angle*dn*dy);
                
                float4 const dataRe = norm_cosine*in_texture.read(uint2(m, y));
                
                Re += (coefRe * dataRe);
                Im += (coefIm * dataRe);
            }
            
            // write into temporary textures
            tmp_texture_ft.write(Re, uint2(2*m+0, n));
            tmp_texture_ft.write(Im, uint2(2*m+1, n));
        }
    }
    
    // row-wise one-dimensional discrete Fourier transform along x-direction
    // exploit symmetry of real dft and calculate reduced number of rows
    for (int dn = 0; dn <= tile_size/2; dn++) {
        for (int dm = 0; dm < tile_size; dm++) {
                       
            int const m = 2*(m0 + dm);
            int const n = n0 + dn;
            int const n2 = n0 + tile_size-dn;
             
            // fill with zeros
            float4 Re1 = zeros;
            float4 Im1 = zeros;
            float4 Re2 = zeros;
            float4 Im2 = zeros;
            
            for (int dx = 0; dx < tile_size; dx++) {
                                  
                int const x = 2*(m0 + dx);
                
                // calculate coefficients
                float const coefRe = cos(angle*dm*dx);
                float const coefIm = sin(angle*dm*dx);
                           
                float4 const dataRe = tmp_texture_ft.read(uint2(x+0, n));
                float4 const dataIm = tmp_texture_ft.read(uint2(x+1, n));
                             
                Re1 += (coefRe*dataRe - coefIm*dataIm);
                Im1 += (coefIm*dataRe + coefRe*dataIm);
                
                Re2 += (coefRe*dataRe + coefIm*dataIm);
                Im2 += (coefIm*dataRe - coefRe*dataIm);
            }
            
            out_texture_ft.write(Re1, uint2(m+0, n));
            out_texture_ft.write(Im1, uint2(m+1, n));
                   
            // exploit symmetry of real dft and calculate values for remaining rows
            if(n2 < n0+tile_size & n2 != n0+tile_size/2)
            {
                out_texture_ft.write(Re2, uint2(m+0, n2));
                out_texture_ft.write(Im2, uint2(m+1, n2));
            }
        }
    }
}


kernel void backward_dft(texture2d<float, access::read> in_texture_ft [[texture(0)]],
                         texture2d<float, access::read_write> tmp_texture_ft [[texture(1)]],
                         texture2d<float, access::write> out_texture [[texture(2)]],
                         constant int& mosaic_pettern_width [[buffer(0)]],
                         constant int& tile_size [[buffer(1)]],
                         constant int& n_textures [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    
    // compute tile positions from gid
    int const m0 = gid.x*tile_size;
    int const n0 = gid.y*tile_size;
    
    // pre-calculate factors for sine and cosine calculation
    float const angle = 2*PI/float(tile_size);
    
    // pre-initalize some vectors
    float4 const zeros       = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 const norm_factor = float4(float(n_textures*tile_size*tile_size), float(n_textures*tile_size*tile_size), float(n_textures*tile_size*tile_size), float(n_textures*tile_size*tile_size));
        
    // row-wise one-dimensional discrete Fourier transform along x-direction
    for (int dn = 0; dn < tile_size; dn++) {
        for (int dm = 0; dm < tile_size; dm++) {
             
            int const m = 2*(m0 + dm);
            int const n = n0 + dn;
            
            // fill with zeros
            float4 Re = zeros;
            float4 Im = zeros;
            
            for (int dx = 0; dx < tile_size; dx++) {
                                  
                int const x = 2*(m0 + dx);
              
                // calculate coefficients
                float const coefRe = cos(angle*dm*dx);
                float const coefIm = sin(angle*dm*dx);
                
                float4 const dataRe = in_texture_ft.read(uint2(x+0, n));
                float4 const dataIm = in_texture_ft.read(uint2(x+1, n));
                
                Re += (coefRe*dataRe - coefIm*dataIm);
                Im += (coefIm*dataRe + coefRe*dataIm);
            }
            
            // write into temporary textures
            tmp_texture_ft.write(Re, uint2(m+0, n));
            tmp_texture_ft.write(Im, uint2(m+1, n));
        }
    }
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (int dm = 0; dm < tile_size; dm++) {
        for (int dn = 0; dn < tile_size; dn++) {
              
            int const m = m0 + dm;
            int const n = n0 + dn;
             
            // fill with zeros
            float4 Re = zeros;
              
            for (int dy = 0; dy < tile_size; dy++) {
                                  
                int const y = n0 + dy;
                
                // calculate coefficients
                float const coefRe = cos(angle*dn*dy);
                float const coefIm = sin(angle*dn*dy);
                           
                float4 const dataRe = tmp_texture_ft.read(uint2(2*m+0, y));
                float4 const dataIm = tmp_texture_ft.read(uint2(2*m+1, y));
                            
                Re += (coefRe*dataRe - coefIm*dataIm);
            }
            
            // normalize result and apply cosine window weight
            Re = Re/norm_factor;
                          
            out_texture.write(Re, uint2(m, n));
        }
    }
}


kernel void merge_frequency_domain(texture2d<float, access::read> aligned_texture [[texture(0)]],
                                   texture2d<float, access::read> diff_texture [[texture(1)]],
                                   texture2d<float, access::read> ref_texture_ft [[texture(2)]],
                                   texture2d<float, access::read_write> tmp_texture_ft [[texture(3)]],
                                   texture2d<float, access::read_write> out_texture_ft [[texture(4)]],
                                   texture2d<float, access::read> rms_texture [[texture(5)]],
                                   texture2d<float, access::read> mismatch_texture [[texture(6)]],
                                   constant float& robustness [[buffer(0)]],
                                   constant int& tile_size [[buffer(1)]],
                                   constant int& tile_size_align [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    
    // derive normalized robustness value: each step yields another factor of two with the idea that the variance of shot noise increases by a factor of two per iso level
    float const robustness_norm = pow(2.0f, (-robustness + 8.0f));
    
    // derive read noise with the idea that read noise increases stronger than a factor of two per iso level to increase noise reduction in darker regions relative to bright regions
    float const read_noise = pow(pow(2.0f, (-robustness + 10.0f)), 1.6);
    
    // combine estimated shot noise and read noise
    float4 const noise_est = rms_texture.read(gid) + read_noise;
    // normalize with tile size and robustness norm
    float4 const noise_norm = noise_est*tile_size*tile_size*robustness_norm;
    
    float4 const mismatch = mismatch_texture.read(gid);
    
    // derive a maximum value for the motion norm with the idea that denoising can be inscreased in static regions with good alignment
    // Google paper: daylight = 1, night = 6, darker night = 14, extreme low-light = 25. We use a linear scaling derived from the robustness value
    // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
    float const max_motion_norm = max(1.0f, pow(1.35f, (12.0f-robustness-1.35f)));
    // increase of noise reduction for small values of mismatch
    float4 const motion_norm = clamp(max_motion_norm-(mismatch-0.02f)*(max_motion_norm-1.0f)/0.15f, 1.0f, max_motion_norm)/sqrt(2.0f);
    //float4 const motion_norm = float4(1.0f, 1.0f, 1.0f, 1.0f);
        
    // compute tile positions from gid
    int const m0 = gid.x*tile_size;
    int const n0 = gid.y*tile_size;

    // pre-calculate factors for sine and cosine calculation
    float const angle = -2*PI/float(tile_size);

    // pre-initalize some vectors
    float4 const zeros = float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    // column-wise one-dimensional discrete Fourier transform along y-direction
    for (int dm = 0; dm < tile_size; dm++) {
        // exploit symmetry of real dft and calculate reduced number of rows
        for (int dn = 0; dn <= tile_size/2; dn++) {
             
            int const m = m0 + dm;
            int const n = n0 + dn;
            
            // fill with zeros
            float4 Re1 = zeros;
            float4 Im1 = zeros;
            
            for (int dy = 0; dy < tile_size; dy++) {
                                  
                int const y = n0 + dy;
            
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts
                float const norm_cosine = (0.5f-0.500f*cos(-angle*(dm+0.5f)))*(0.5f-0.500f*cos(-angle*(dy+0.5f)));
                // calculate modified raised cosine window weight for blending tiles to suppress artifacts (slightly adapted compared to original publication)
                //float const norm_cosine = (0.5f-0.505f*cos(-angle*(dm+0.5f)))*(0.5f-0.505f*cos(-angle*(dy+0.5f)));
                                
                // calculate coefficients
                float const coefRe = cos(angle*dn*dy);
                float const coefIm = sin(angle*dn*dy);
                
                float4 const dataRe = norm_cosine*aligned_texture.read(uint2(m, y));
                
                Re1 += (coefRe * dataRe);
                Im1 += (coefIm * dataRe);
            }
            
            // write into temporary textures
            tmp_texture_ft.write(Re1, uint2(2*m+0, n));
            tmp_texture_ft.write(Im1, uint2(2*m+1, n));
        }
    }
    
    // row-wise one-dimensional discrete Fourier transform along x-direction
    // exploit symmetry of real dft and calculate reduced number of rows
    for (int dn = 0; dn <= tile_size/2; dn++) {
        for (int dm = 0; dm < tile_size; dm++) {
                       
            int const m = 2*(m0 + dm);
            int const n = n0 + dn;
            int const n2 = n0 + tile_size-dn;
             
            float4 Re1 = zeros;
            float4 Im1 = zeros;
            float4 Re2 = zeros;
            float4 Im2 = zeros;
            
            for (int dx = 0; dx < tile_size; dx++) {
                                  
                int const x = 2*(m0 + dx);
                
                // calculate coefficients
                float const coefRe = cos(angle*dm*dx);
                float const coefIm = sin(angle*dm*dx);
                           
                float4 const dataRe = tmp_texture_ft.read(uint2(x+0, n));
                float4 const dataIm = tmp_texture_ft.read(uint2(x+1, n));
                            
                Re1 += (coefRe*dataRe - coefIm*dataIm);
                Im1 += (coefIm*dataRe + coefRe*dataIm);
                      
                Re2 += (coefRe*dataRe + coefIm*dataIm);
                Im2 += (coefIm*dataRe - coefRe*dataIm);
            }
   
            float4 Re3 = ref_texture_ft.read(uint2(m+0, n));
            float4 Im3 = ref_texture_ft.read(uint2(m+1, n));
            
            // calculation of merging weight by Wiener shrinkage
            float4 weight4 = (Re3-Re1)*(Re3-Re1) + (Im3-Im1)*(Im3-Im1);
            weight4 = weight4/(weight4 + motion_norm*noise_norm);
            
            // use the same weight for all color channels to reduce color artifacts. Use maximum of weight4 as this increases motion robustness.
            // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
            float weight = clamp(max(weight4[0], max(weight4[1], max(weight4[2], weight4[3]))), 0.0f, 1.0f);
                        
            // merging of two textures
            float4 merged_Re = out_texture_ft.read(uint2(m+0, n)) + (1.0f-weight)*Re1 + weight*Re3;
            float4 merged_Im = out_texture_ft.read(uint2(m+1, n)) + (1.0f-weight)*Im1 + weight*Im3;
         
            out_texture_ft.write(merged_Re, uint2(m+0, n));
            out_texture_ft.write(merged_Im, uint2(m+1, n));
            
            // exploit symmetry of real dft and calculate values for remaining rows
            if(n2 < n0+tile_size & n2 != n0+tile_size/2)
            {
                Re3 = ref_texture_ft.read(uint2(m+0, n2));
                Im3 = ref_texture_ft.read(uint2(m+1, n2));
                
                // calculation of merging weight by Wiener shrinkage
                weight4 = (Re3-Re2)*(Re3-Re2) + (Im3-Im2)*(Im3-Im2);
                weight4 = weight4/(weight4 + motion_norm*noise_norm);
                
                // use the same weight for all color channels to reduce color artifacts. Use maximum of weight4 as this increases motion robustness.
                // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
                weight = clamp(max(weight4[0], max(weight4[1], max(weight4[2], weight4[3]))), 0.0f, 1.0f);
                            
                // merging of two textures
                merged_Re = out_texture_ft.read(uint2(m+0, n2)) + (1.0f-weight)*Re2 + weight*Re3;
                merged_Im = out_texture_ft.read(uint2(m+1, n2)) + (1.0f-weight)*Im2 + weight*Im3;
              
                out_texture_ft.write(merged_Re, uint2(m+0, n2));
                out_texture_ft.write(merged_Im, uint2(m+1, n2));
            }
        }
    }
}


kernel void calculate_mismatch_rgba(texture2d<float, access::read> ref_texture [[texture(0)]],
                                    texture2d<float, access::read> aligned_texture [[texture(1)]],
                                    texture2d<float, access::read> rms_texture [[texture(2)]],
                                    texture2d<float, access::write> mismatch_texture [[texture(3)]],
                                    constant int& tile_size [[buffer(0)]],
                                    constant int& tile_size_align [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
        
    // compute tile positions from gid
    int const m0 = gid.x*tile_size;
    int const n0 = gid.y*tile_size;
    
    // use only estimated shot noise here
    float4 const noise_est = rms_texture.read(gid);
    
    float const angle = 2*PI/float(tile_size);
    
    // estimate motion mismatch as the absolute difference of reference tile and comparison tile
    // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
    // use 16x16, 32x32 or 64x64 tile size here as used for alignment
    
    int const x_start = max(0, m0 + tile_size/2 - tile_size_align/2);
    int const y_start = max(0, n0 + tile_size/2 - tile_size_align/2);
    
    int const x_end = min(int(ref_texture.get_width()-1),  m0 + tile_size/2 + tile_size_align/2);
    int const y_end = min(int(ref_texture.get_height()-1), n0 + tile_size/2 + tile_size_align/2);
    
    float tile_diff = 0.f;
    float n_total = 0.f;
     
    for (int dy = y_start; dy < y_end; dy++) {
        for (int dx = x_start; dx < x_end; dx++) {
        
            float4 const diff = float4(ref_texture.read(uint2(dx, dy))) - float4(aligned_texture.read(uint2(dx, dy)));
             
            // calculate modified raised cosine window weight for blending tiles to suppress artifacts
            float const weight = (0.5f-0.500f*cos(angle*(dx+0.5f)))*(0.5f-0.500f*cos(angle*(dy+0.5f)));
            // calculate modified raised cosine window weight for blending tiles to suppress artifacts (slightly adapted compared to original publication)
            //float const weight = (0.5f-0.505f*cos(angle*(dx+0.5f)))*(0.5f-0.505f*cos(angle*(dy+0.5f)));
                 
            tile_diff += weight*(0.25f * abs(diff[0]+diff[1]+diff[2]+diff[3]));
             
            n_total += weight;
        }
    }
     
    tile_diff /= float(n_total);
    
    /* // potential additional correction of artifacts around highlights
    float tile_diff = 0.f;
    float tile_int1 = 0.f;
    float tile_int2 = 0.f;
    float tile_int3 = 0.f;
    float tile_int4 = 0.f;
    
    for (int dy = 0; dy < tile_size/2; dy++) {
        for (int dx = 0; dx < tile_size/2; dx++) {
            
            int const x = m0 + dx;
            int const y = n0 + dy;
            
            float4 const ref_val = ref_texture.read(uint2(x, y));
            float4 const diff = ref_val - aligned_texture.read(uint2(x, y));
               
            tile_int1 += (0.25f * (ref_val[0]+ref_val[1]+ref_val[2]+ref_val[3]));
            tile_diff += (0.25f * abs(diff[0]+diff[1]+diff[2]+diff[3]));
        }
    }
    
    for (int dy = 0; dy < tile_size/2; dy++) {
        for (int dx = tile_size/2; dx < tile_size; dx++) {
            
            int const x = m0 + dx;
            int const y = n0 + dy;
            
            float4 const ref_val = ref_texture.read(uint2(x, y));
            float4 const diff = ref_val - aligned_texture.read(uint2(x, y));
               
            tile_int2 += (0.25f * (ref_val[0]+ref_val[1]+ref_val[2]+ref_val[3]));
            tile_diff += (0.25f * abs(diff[0]+diff[1]+diff[2]+diff[3]));
        }
    }

    for (int dy = tile_size/2; dy < tile_size; dy++) {
        for (int dx = 0; dx < tile_size/2; dx++) {
            
            int const x = m0 + dx;
            int const y = n0 + dy;
            
            float4 const ref_val = ref_texture.read(uint2(x, y));
            float4 const diff = ref_val - aligned_texture.read(uint2(x, y));
               
            tile_int3 += (0.25f * (ref_val[0]+ref_val[1]+ref_val[2]+ref_val[3]));
            tile_diff += (0.25f * abs(diff[0]+diff[1]+diff[2]+diff[3]));
        }
    }
    
    for (int dy = tile_size/2; dy < tile_size; dy++) {
        for (int dx = tile_size/2; dx < tile_size; dx++) {
            
            int const x = m0 + dx;
            int const y = n0 + dy;
            
            float4 const ref_val = ref_texture.read(uint2(x, y));
            float4 const diff = ref_val - aligned_texture.read(uint2(x, y));
               
            tile_int4 += (0.25f * (ref_val[0]+ref_val[1]+ref_val[2]+ref_val[3]));
            tile_diff += (0.25f * abs(diff[0]+diff[1]+diff[2]+diff[3]));
        }
    }
        
    tile_diff /= float(tile_size*tile_size);
    
    float const min_int = min(tile_int1, min(tile_int2, min(tile_int3, tile_int4)));
    float const max_int = max(tile_int1, max(tile_int2, max(tile_int3, tile_int4)));
    
    float const ratio = max_int/min_int;
    float factor = 1.0f;
    
    if(ratio > 2.0f)
    {
        //factor = min(sqrt(ratio-1.0f), 10.0f);
    }
    */
    // calculation of mismatch ratio
    //float4 const mismatch4 = (tile_diff*tile_diff) / (tile_diff*tile_diff + noise_est);
    //float4 const mismatch4 = (tile_diff*tile_diff) / (noise_est+1e-12);
    //float4 const mismatch4 = tile_diff / (factor*sqrt(noise_est+1e-12));
    float4 const mismatch4 = tile_diff / sqrt(noise_est+1e-12);
    float const mismatch = 0.25f*(mismatch4[0] + mismatch4[1] + mismatch4[2] + mismatch4[3]);
    
    mismatch_texture.write(mismatch, gid);
}


kernel void normalize_mismatch(texture2d<float, access::read_write> mismatch_texture [[texture(0)]],
                               constant float* mean_mismatch_buffer [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
    
    // load args
    float const mean_mismatch = mean_mismatch_buffer[0];
    
    float mismatch_norm = mismatch_texture.read(gid).r;
    
    mismatch_norm /= (mean_mismatch*5.0f + 1e-12f);
    
    mismatch_norm = clamp(mismatch_norm, 0.0f, 1.0f);
    
    mismatch_texture.write(mismatch_norm, gid);
}
