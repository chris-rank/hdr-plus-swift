import Foundation
import MetalKit
import MetalPerformanceShaders


// possible error types during the alignment
enum AlignmentError: Error {
    case less_than_two_images
    case inconsistent_extensions
    case inconsistent_resolutions
    case conversion_failed
    case missing_dng_converter
}


// all the relevant information about image tiles in a single struct
struct TileInfo {
    var tile_size: Int
    var tile_size_merge: Int
    var search_dist: Int
    var n_tiles_x: Int
    var n_tiles_y: Int
    var n_pos_1d: Int
    var n_pos_2d: Int
}

struct DirectionalPadValues {
    let left:   Int
    let right:  Int
    let top:    Int
    let bottom: Int
}

struct DimensionalPadValues {
    let x: Int
    let y: Int
}

/**
 * Buffer textures used by the frequency merge algorithm packaged together for readability and compactness.
 * These buffers are used so that they can be allocated onces at the beggining of the merging sequence and then re-used, significantly reducing the amount of memory that gets allocated & deallocated as the application runs.
 */
struct FrequencyMergeTextureBuffers {
    // *************************************
    // FREQUENCY (RGBA 32bit) TEXTURES
    //
    // The frequency textures have half-height but full width versus the Bayer texture.
    //      The height is halved since it takes the 2x2 and stacks it vertically from a "r" format into an "rgba" format.
    //      The width is NOT halved since both the Real and Imaginary components are stored in the same texture. Data is stored Real first followed by the Imaginary component
    // *************************************
    //
    // Accumulator for the storing the Real and Imaginary components from the Fourier transform as each imagine is merged with the reference
    let final_frequency_texture: MTLTexture
    // The Real and Imaginary component of the Fourier transform of the reference texture
    let ref_frequency_texture: MTLTexture
    // The Real and Imaginary component of the Fourier transform of the current texture being merged AFTER being aligned
    let aligned_frequency_texture: MTLTexture
    // Temporary buffer used by the Fourier transform implementation to store Real and Imagine components
    let tmp_frequency_texture: MTLTexture
    
    // *************************************
    // 4-CHANNEL (RGBA) TEXTURES
    // *************************************
    //
    // RGBA version of the reference texture, needed since fourier transform needs rgba format
    let ref_rgba_texture: MTLTexture
    // RGBA version of the current texture being merged in AFTER it's been aligned
    let aligned_rgba_texture: MTLTexture
    // Texture to store the Root-mean-square value for each color
    
    // *************************************
    // SINGLE-CHANNEL (R 32-BIT) TEXTURES
    // *************************************
    //
    // Padded reference texture in RGGB format
    let ref_texture: MTLTexture
    // Padded comparison texture in RGGB format BEFORE aligning
    let comp_texture: MTLTexture
    // Padded comparison texture in RGGB format AFTER aligning
    let aligned_texture: MTLTexture
    
    // *************************************
    // ALIGNMENT PYRAMID TEXTURES
    // *************************************
    // Pyramid for the reference texture
    let ref_pyramid: Array<MTLTexture>
    // Pyramid for the comparison texture, overriden when each image is compared
    let comp_pyramid: Array<MTLTexture>
}

struct PyramidInfo {
    let tile_sizes: Array<Int>
    let search_dist: Array<Int>
    let downscale_factors: Array<Int>
    let tile_factor: Int
}

// class to store the progress of the align+merge
class ProcessingProgress: ObservableObject {
    @Published var int = 0
    @Published var includes_conversion = false
}


// set up Metal device
let device = MTLCreateSystemDefaultDevice()!
let command_queue = device.makeCommandQueue()!
let mfl = device.makeDefaultLibrary()!


// compile metal functions / pipelines
let fill_with_zeros_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "fill_with_zeros")!)
let texture_uint16_to_float_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "texture_uint16_to_float")!)
let convert_uint16_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "convert_uint16")!)
let upsample_nearest_int_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "upsample_nearest_int")!)
let upsample_bilinear_float_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "upsample_bilinear_float")!)
let avg_pool_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "avg_pool")!)
let blur_mosaic_x_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "blur_mosaic_x")!)
let blur_mosaic_y_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "blur_mosaic_y")!)
let extend_texture_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "extend_texture")!)
let crop_texture_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "crop_texture")!)
let add_texture_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "add_texture")!)
let add_crop_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "add_crop")!)
let copy_texture_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "copy_texture")!)
let convert_rgba_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "convert_rgba")!)
let convert_bayer_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "convert_bayer")!)
let average_x_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "average_x")!)
let average_y_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "average_y")!)
let average_x_rgba_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "average_x_rgba")!)
let average_y_rgba_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "average_y_rgba")!)
let compute_tile_differences_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "compute_tile_differences")!)
let compute_tile_differences25_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "compute_tile_differences25")!)
let compute_tile_alignments_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "compute_tile_alignments")!)
let correct_upsampling_error_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "correct_upsampling_error")!)
let warp_texture_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "warp_texture")!)
let add_textures_weighted_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "add_textures_weighted")!)
let color_difference_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "color_difference")!)
let compute_merge_weight_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "compute_merge_weight")!)
let calculate_rms_rgba_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "calculate_rms_rgba")!)
let forward_dft_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "forward_dft")!)
let forward_fft_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "forward_fft")!)
let backward_dft_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "backward_dft")!)
let backward_fft_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "backward_fft")!)
let merge_frequency_domain_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "merge_frequency_domain")!)
let calculate_mismatch_rgba_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "calculate_mismatch_rgba")!)
let normalize_mismatch_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "normalize_mismatch")!)
let correct_hotpixels_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "correct_hotpixels")!)
let identify_hotpixels_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "identify_hotpixels")!)
let deconvolute_frequency_domain_state = try! device.makeComputePipelineState(function: mfl.makeFunction(name: "deconvolute_frequency_domain")!)



// ===========================================================================================================
// Main functions
// ===========================================================================================================


// main function of the burst photo app
func perform_denoising(image_urls: [URL], progress: ProcessingProgress, ref_idx: Int = 0, merging_algorithm: String = "Better speed", tile_size: Int = 32, kernel_size: Int = 5, noise_reduction: Double = 13.0) throws -> URL {
    
    // measure execution time
    let t0 = DispatchTime.now().uptimeNanoseconds
    var t = t0
    
    // check that all images are of the same extension
    let image_extension = image_urls[0].pathExtension
    let all_extensions_same = image_urls.allSatisfy{$0.pathExtension == image_extension}
    if !all_extensions_same {throw AlignmentError.inconsistent_extensions}
    
    // check that 2+ images were provided
    let n_images = image_urls.count
    if n_images < 2 {throw AlignmentError.less_than_two_images}
    
    // create output directory
    let out_dir = NSHomeDirectory() + "/Pictures/Burst Photo/"
    if !FileManager.default.fileExists(atPath: out_dir) {
        try FileManager.default.createDirectory(atPath: out_dir, withIntermediateDirectories: true, attributes: nil)
    }
    
    // create a directory for temporary dngs inside the output directory
    let tmp_dir = out_dir + ".dngs/"
    try FileManager.default.createDirectory(atPath: tmp_dir, withIntermediateDirectories: true)
       
    // ensure that all files are .dng, converting them if necessary
    var dng_urls = image_urls
    let convert_to_dng = image_extension.lowercased() != "dng"
    DispatchQueue.main.async { progress.includes_conversion = convert_to_dng }
    let dng_converter_path = "/Applications/Adobe DNG Converter.app"
    let final_dng_conversion = FileManager.default.fileExists(atPath: dng_converter_path)
    
    if convert_to_dng {
        // check if dng converter is installed
        if !FileManager.default.fileExists(atPath: dng_converter_path) {
            // if dng coverter is not installed, prompt user
            throw AlignmentError.missing_dng_converter
        } else {
            // the dng converter is installed -> use it
            dng_urls = try convert_images_to_dng(image_urls, dng_converter_path, tmp_dir)
            print("Time to convert images: ", Float(DispatchTime.now().uptimeNanoseconds - t) / 1_000_000_000)
            DispatchQueue.main.async { progress.int += 10000000 }
            t = DispatchTime.now().uptimeNanoseconds
        }
    }
    
    // set output location
    let in_url = dng_urls[ref_idx]
    let in_filename = in_url.deletingPathExtension().lastPathComponent
    // the value of the noise reduction strength is written into the filename    
    let suffix_merging =  merging_algorithm=="Better speed" ? "s" : "q"
    let out_filename = in_filename + (noise_reduction==23.0 ? "_merged_avg.dng" : "_merged_" + suffix_merging + "\(Int(noise_reduction+0.5)).dng")
    let out_path = (final_dng_conversion ? tmp_dir : out_dir) + out_filename
    var out_url = URL(fileURLWithPath: out_path)
    
    // load images
    t = DispatchTime.now().uptimeNanoseconds
    var (textures, mosaic_pattern_width) = try load_images(dng_urls)
    print("Time to load all images: ", Float(DispatchTime.now().uptimeNanoseconds - t) / 1_000_000_000)
    t = DispatchTime.now().uptimeNanoseconds
    DispatchQueue.main.async { progress.int += (convert_to_dng ? 10000000 : 20000000) }
    
    // convert images from uint16 to float16
    textures = textures.map{texture_uint16_to_float($0)}
     
    // set the maximum resolution of the smallest pyramid layer
    // Low: 128
    // Medium: 64
    // High: 32
    let search_distance_int = 64
    
    // use a 32 bit float as final image
    let final_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: textures[ref_idx].width, height: textures[ref_idx].height, mipmapped: false)
    final_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let final_texture = device.makeTexture(descriptor: final_texture_descriptor)!
    fill_with_zeros(final_texture)
    
    // special mode: simple temporal averaging without alignment and robust merging
    if noise_reduction == 23.0 {
        print("Special mode: temporal averaging only...")
         
        // correction of hot pixels
        if mosaic_pattern_width == 2 {
            correct_hotpixels(textures)
        }
                  
        // iterate over all images
        for comp_idx in 0..<image_urls.count {
            
            add_texture(textures[comp_idx], to: final_texture, image_urls.count)
            DispatchQueue.main.async { progress.int += Int(80000000/Double(image_urls.count)) }
        }
        

    // sophisticated approach with alignment of tiles and merging of tiles in frequency domain (only 2x2 Bayer pattern)
    } else if mosaic_pattern_width == 2 && merging_algorithm == "Better quality" {
        print("Merging in the frequency domain...")
        
        // the tile size for merging in frequency domain is set to 8x8 for all tile sizes used for alignment. The smaller tile size leads to a reduction of artifacts at specular highlights at the expense of a slightly reduced suppression of low-frequency noise in the shadows
        // see https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf for more details
        let tile_size_merge = Int(8)
          
        // derive normalized robustness value: two steps in noise_reduction (-2.0 in this case) yield an increase by a factor of two in the robustness norm with the idea that the variance of shot noise increases by a factor of two per iso level
        let robustness_rev = 0.5*(26.5-Double(Int(noise_reduction+0.5)))
        let robustness_norm = pow(2.0, (-robustness_rev + 7.5));
        
        // derive estimate of read noise with the idea that read noise increases approx. by a factor of three (stronger than increase in shot noise) per iso level to increase noise reduction in darker regions relative to bright regions
        let read_noise = pow(pow(2.0, (-robustness_rev + 10.0)), 1.6);
        
        // derive a maximum value for the motion norm with the idea that denoising can be stronger in static regions with good alignment compared to regions with motion
        // factors from Google paper: daylight = 1, night = 6, darker night = 14, extreme low-light = 25. We use a continuous value derived from the robustness value to cover a similar range as proposed in the paper
        // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
        let max_motion_norm = max(1.0, pow(1.3, (11.0-robustness_rev)));
        
        // set mode for Fourier transformations ("DFT" or "FFT")
        // for the mode "FFT", only tile sizes <= 16 are possible due to some restrictions within the function
        let ft_mode = (tile_size_merge <= Int(16) ? "FFT" : "DFT")
        
        // correction of hot pixels
        correct_hotpixels(textures)
        
        let pyramid_info = create_downscaling_pyramid_info(ref_texture: textures[ref_idx], mosaic_pattern_width: mosaic_pattern_width, tile_size: tile_size, search_distance: search_distance_int)
        
        let (buffers, crop_merge, pad_align, tile_info_merge) = create_frequency_buffers(ref_texture: textures[ref_idx], mosaic_pattern_width: mosaic_pattern_width, search_distance: search_distance_int, tile_size: tile_size, tile_size_merge: tile_size_merge, pyramid_info: pyramid_info)
        
        // perform align and merge 4 times in a row with slight displacement of the frame to prevent artifacts in the merging process. The shift equals the tile size used in the merging process here, which later translates into tile_size_merge/2 when each color channel is processed independently
        try align_merge_frequency_domain(progress: progress, shift_left_not_right: true, shift_top_not_bottom: true, ref_idx: ref_idx, search_distance: search_distance_int, robustness_norm: robustness_norm, read_noise: read_noise, max_motion_norm: max_motion_norm, ft_mode: ft_mode, textures: textures, final_texture: final_texture, tile_info_merge: tile_info_merge, buffers: buffers, pad_align: pad_align, crop_merge: crop_merge, pyramid_info: pyramid_info)
        
        try align_merge_frequency_domain(progress: progress, shift_left_not_right: false, shift_top_not_bottom: true, ref_idx: ref_idx, search_distance: search_distance_int, robustness_norm: robustness_norm, read_noise: read_noise, max_motion_norm: max_motion_norm, ft_mode: ft_mode, textures: textures, final_texture: final_texture, tile_info_merge: tile_info_merge, buffers: buffers, pad_align: pad_align, crop_merge: crop_merge, pyramid_info: pyramid_info)
        
        try align_merge_frequency_domain(progress: progress, shift_left_not_right: true, shift_top_not_bottom: false, ref_idx: ref_idx, search_distance: search_distance_int, robustness_norm: robustness_norm, read_noise: read_noise, max_motion_norm: max_motion_norm, ft_mode: ft_mode, textures: textures, final_texture: final_texture, tile_info_merge: tile_info_merge, buffers: buffers, pad_align: pad_align, crop_merge: crop_merge, pyramid_info: pyramid_info)
        
        try align_merge_frequency_domain(progress: progress, shift_left_not_right: false, shift_top_not_bottom: false, ref_idx: ref_idx, search_distance: search_distance_int, robustness_norm: robustness_norm, read_noise: read_noise, max_motion_norm: max_motion_norm, ft_mode: ft_mode, textures: textures, final_texture: final_texture, tile_info_merge: tile_info_merge, buffers: buffers, pad_align: pad_align, crop_merge: crop_merge, pyramid_info: pyramid_info)
        
    // sophisticated approach with alignment of tiles and merging of tiles in the spatial domain (when pattern is not 2x2 Bayer)
    } else {
        print("Merging in the spatial domain...")
    
        let kernel_size = Int(16) // kernel size of binomial filtering used for blurring the image
        
        // derive normalized robustness value: four steps in noise_reduction (-4.0 in this case) yield an increase by a factor of two in the robustness norm with the idea that the sd of shot noise increases by a factor of sqrt(2) per iso level
        let robustness_rev = 0.5*(26.0-Double(Int(noise_reduction+0.5)))
        let robustness_norm = 0.15*pow(sqrt(2), robustness_rev) + 0.2
        
        let pyramid_info = create_downscaling_pyramid_info(ref_texture: textures[ref_idx], mosaic_pattern_width: mosaic_pattern_width, tile_size: tile_size, search_distance: search_distance_int)
        
        let texture_width_orig = textures[ref_idx].width
        let texture_height_orig = textures[ref_idx].height
        
        var _pad_align_x = Int(ceil(Float(texture_width_orig)/Float(pyramid_info.tile_factor)))
        _pad_align_x = (_pad_align_x*Int(pyramid_info.tile_factor) - texture_width_orig)/2
        
        var _pad_align_y = Int(ceil(Float(texture_height_orig)/Float(pyramid_info.tile_factor)))
        _pad_align_y = (_pad_align_y*Int(pyramid_info.tile_factor) - texture_height_orig)/2
        
        let pad_align = DimensionalPadValues(x: _pad_align_x, y: _pad_align_y)
        
        let (ref_pyramid, comp_pyramid) = generate_pyramid_buffers(with_metadata_from: textures[ref_idx], with_pyramid_info: pyramid_info, pad_align: pad_align, tile_size: tile_size)
        
        try align_merge_spatial_domain(progress: progress, ref_idx: ref_idx, mosaic_pattern_width: mosaic_pattern_width, search_distance: search_distance_int, tile_size: tile_size, kernel_size: kernel_size, robustness: robustness_norm, textures: textures, final_texture: final_texture, ref_pyramid, comp_pyramid, pyramid_info: pyramid_info)
    }
    
    // convert final image to 16 bit integer
    let output_texture_uint16 = convert_uint16(final_texture)
      
    print("Time to align+merge all images: ", Float(DispatchTime.now().uptimeNanoseconds - t) / 1_000_000_000)
    t = DispatchTime.now().uptimeNanoseconds
    
    DispatchQueue.main.async { progress.int += 10000000 }
    
    // save the output image
    try texture_to_dng(output_texture_uint16, in_url, out_url)
    
    // check if dng converter is installed
    if final_dng_conversion {
        let path_delete = out_dir + out_filename
          
        // delete dng file if an old version exists
        if FileManager.default.fileExists(atPath: path_delete) {
            try FileManager.default.removeItem(atPath: path_delete)
        }
        
        // the dng converter is installed -> convert output DNG saved before with Adobe DNG Converter, which increases compatibility of the resulting DNG
        let final_url = try convert_images_to_dng([out_url], dng_converter_path, out_dir)
                   
        // update out URL to new file
        out_url = final_url[0]
    }
    
    // delete the temporary dng directory
    try FileManager.default.removeItem(atPath: tmp_dir)
    
    print("Time to save final image: ", Float(DispatchTime.now().uptimeNanoseconds - t) / 1_000_000_000)
    print("------------------------------------------------")
    print("Total processing time for", textures.count, "images: ", Float(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000_000)
    print("------------------------------------------------")
    
    return out_url
}


// convenience function for the spatial merging approach
func align_merge_spatial_domain(progress: ProcessingProgress, ref_idx: Int, mosaic_pattern_width: Int, search_distance: Int, tile_size: Int, kernel_size: Int, robustness: Double, textures: [MTLTexture], final_texture: MTLTexture, _ ref_pyramid: Array<MTLTexture>, _ comp_pyramid: Array<MTLTexture>, pyramid_info: PyramidInfo) throws {
    
    // set original texture size
    let texture_width_orig = textures[ref_idx].width
    let texture_height_orig = textures[ref_idx].height
    
    // calculate padding for extension of the image frame with zeros
    // For the alignment, the frame may be extended further by pad_align due to the following reason: the alignment is performed on different resolution levels and alignment vectors are upscaled by a simple multiplication by 2. As a consequence, the frame at all resolution levels has to be a multiple of the tile sizes of these resolution levels.
    var _pad_align_x = Int(ceil(Float(texture_width_orig)/Float(pyramid_info.tile_factor)))
    _pad_align_x = (_pad_align_x*Int(pyramid_info.tile_factor) - texture_width_orig)/2
    var _pad_align_y = Int(ceil(Float(texture_height_orig)/Float(pyramid_info.tile_factor)))
    _pad_align_y = (_pad_align_y*Int(pyramid_info.tile_factor) - texture_height_orig)/2
    
    let pad_align = DimensionalPadValues(x: _pad_align_x, y: _pad_align_y)
    
    let comp_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float,
                                                                           width: textures[ref_idx].width+2*pad_align.x,
                                                                           height: textures[ref_idx].height+2*pad_align.y,
                                                                          mipmapped: false)
    comp_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let comp_texture = device.makeTexture(descriptor: comp_texture_descriptor)!
    fill_with_zeros(comp_texture)
    
    let ref_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float,
                                                                          width: textures[ref_idx].width+2*pad_align.x,
                                                                          height: textures[ref_idx].height+2*pad_align.y,
                                                                          mipmapped: false)
    ref_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let ref_texture = device.makeTexture(descriptor: ref_texture_descriptor)!
    fill_with_zeros(ref_texture)
    
    // set reference texture
    extend_texture(textures[ref_idx], onto: ref_texture, pad_align.x, pad_align.y)
    
    let aligned_texture_uncropped_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: ref_texture.pixelFormat, width: ref_texture.width, height: ref_texture.height, mipmapped: false)
    aligned_texture_uncropped_descriptor.usage = [.shaderRead, .shaderWrite]
    let aligned_texture_uncropped = device.makeTexture(descriptor: aligned_texture_uncropped_descriptor)!
    
    // build reference pyramid
    build_pyramid(for_texture: ref_texture, save_in: ref_pyramid, pyramid_info.downscale_factors)
    
    // blur reference texure and estimate noise standard deviation
    // -  the computation is done here to avoid repeating the same computation in 'robust_merge()'
    let ref_texture_blurred = blur_mosaic_texture(textures[ref_idx], kernel_size, mosaic_pattern_width)
    let noise_sd = estimate_color_noise(textures[ref_idx], ref_texture_blurred, mosaic_pattern_width)

    // iterate over comparison images
    for comp_idx in 0..<textures.count {
        
        // add the reference texture to the output
        if comp_idx == ref_idx {
            add_texture(textures[comp_idx], to: final_texture, textures.count)
            DispatchQueue.main.async { progress.int += Int(80000000/Double(textures.count)) }
            continue
        }
        
        // check that the comparison image has the same resolution as the reference image
        if (textures[comp_idx].width != textures[comp_idx].width) || (textures[comp_idx].height != textures[comp_idx].height) {
            throw AlignmentError.inconsistent_resolutions
        }
        
        // set comparison texture
        extend_texture(textures[comp_idx], onto: comp_texture, pad_align.x, pad_align.y)
       
        // align comparison texture
        align_texture(ref_pyramid, comp_pyramid, comp_texture, save_in: aligned_texture_uncropped, pyramid_info)
        let aligned_texture = crop_texture(aligned_texture_uncropped, pad_align.x, pad_align.x, pad_align.y, pad_align.y)
        
        // robust-merge the texture
        let merged_texture = robust_merge(textures[ref_idx], ref_texture_blurred, aligned_texture, kernel_size, robustness, noise_sd, mosaic_pattern_width)
        
        // add robust-merged texture to the output image
        add_texture(merged_texture, to: final_texture, textures.count)
        
        // sync GUI progress
        DispatchQueue.main.async { progress.int += Int(80000000/Double(textures.count)) }
    }
}

// convenience function for the frequency-based merging approach
func align_merge_frequency_domain(progress: ProcessingProgress, shift_left_not_right: Bool, shift_top_not_bottom: Bool, ref_idx: Int, search_distance: Int, robustness_norm: Double, read_noise: Double, max_motion_norm: Double, ft_mode: String, textures: [MTLTexture], final_texture: MTLTexture, tile_info_merge: TileInfo, buffers: FrequencyMergeTextureBuffers, pad_align: DimensionalPadValues, crop_merge: DimensionalPadValues, pyramid_info: PyramidInfo) throws {

    let shift = DirectionalPadValues(left:   shift_left_not_right ? tile_info_merge.tile_size_merge : 0,
                                     right:  shift_left_not_right ? 0                               : tile_info_merge.tile_size_merge,
                                     top:    shift_top_not_bottom ? tile_info_merge.tile_size_merge : 0,
                                     bottom: shift_top_not_bottom ? 0                               : tile_info_merge.tile_size_merge)
    
    // add shifts for artifact suppression
    let pad = DirectionalPadValues(left:   pad_align.x + shift.left,
                                   right:  pad_align.x + shift.right,
                                   top:    pad_align.y + shift.top,
                                   bottom: pad_align.y + shift.bottom)
    
    // set and extend reference texture
    extend_texture(textures[ref_idx], onto: buffers.ref_texture, pad.left, pad.top)
        
    // build reference pyramid
    build_pyramid(for_texture: buffers.ref_texture, save_in: buffers.ref_pyramid, pyramid_info.downscale_factors)
    
    // convert reference texture into RGBA pixel format that SIMD instructions can be applied
    convert_rgba(buffers.ref_texture, save_in: buffers.ref_rgba_texture, crop_merge.x, crop_merge.y)
    
    // estimate noise level of tiles
    let rms_texture = calculate_rms_rgba(buffers.ref_rgba_texture, tile_info_merge)
    
    // generate texture to accumulate the total mismatch
    let total_mismatch_texture = texture_like(rms_texture)
    fill_with_zeros(total_mismatch_texture)
    
    // transform reference texture into the frequency domain
    forward_ft(buffers.ref_rgba_texture, save_in: buffers.ref_frequency_texture, buffers.tmp_frequency_texture, tile_info_merge, mode: ft_mode)
    
    // add reference texture to the final texture
    copy_texture(buffers.ref_frequency_texture, onto: buffers.final_frequency_texture)
    
    // iterate over comparison images
    for comp_idx in 0..<textures.count {
        let t0 = DispatchTime.now().uptimeNanoseconds
        
        // if comparison image is equal to reference image, do nothing
        if comp_idx == ref_idx {
            continue
        }
         
        // set and extend comparison texture
        extend_texture(textures[comp_idx], onto: buffers.comp_texture, pad.left, pad.top)
        
        // align comparison texture
        align_texture(buffers.ref_pyramid, buffers.comp_pyramid, buffers.comp_texture, save_in: buffers.aligned_texture, pyramid_info)
        convert_rgba(buffers.aligned_texture, save_in: buffers.aligned_rgba_texture, crop_merge.x, crop_merge.y)
                 
        // calculate mismatch texture
        let mismatch_texture = calculate_mismatch_rgba(buffers.aligned_rgba_texture, buffers.ref_rgba_texture, rms_texture, tile_info_merge)

        // normalize mismatch texture
        let mean_mismatch = texture_mean(crop_texture(mismatch_texture, shift.left/tile_info_merge.tile_size_merge, shift.right/tile_info_merge.tile_size_merge, shift.top/tile_info_merge.tile_size_merge, shift.bottom/tile_info_merge.tile_size_merge), "r")
        normalize_mismatch(mismatch_texture, mean_mismatch)
           
        // add mismatch texture to the total, accumulated mismatch texture
        add_texture(mismatch_texture, to: total_mismatch_texture, textures.count)
        
        // start debug capture
        //let capture_manager = MTLCaptureManager.shared()
        //let capture_descriptor = MTLCaptureDescriptor()
        //capture_descriptor.captureObject = device
        //try! capture_manager.startCapture(with: capture_descriptor)
          
        // transform aligned comparison texture into the frequency domain
        forward_ft(buffers.aligned_rgba_texture, save_in: buffers.aligned_frequency_texture, buffers.tmp_frequency_texture, tile_info_merge, mode: ft_mode)
        
        // merge aligned comparison texture with reference texture in the frequency domain
        merge_frequency_domain(buffers.ref_frequency_texture, buffers.aligned_frequency_texture, buffers.final_frequency_texture, rms_texture, mismatch_texture, robustness_norm, read_noise, max_motion_norm, tile_info_merge);
               
        // stop debug capture
        //capture_manager.stopCapture()
      
        // sync GUI progress
        print("Align+merge: ", Float(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000_000)
        DispatchQueue.main.async { progress.int += Int(80000000/Double(4*(textures.count-1))) }
    }
    
    // apply simple deconvolution to slightly correct potential blurring from misalignment of bursts
    deconvolute_frequency_domain(buffers.final_frequency_texture, total_mismatch_texture, tile_info_merge)
    
    // Transform output texture back to image domain
    // NOTE: Re-using `aligned_rgba_texture` since it is the correct size, saves us from alocating another texture
    backward_ft(buffers.final_frequency_texture, save_in: buffers.aligned_rgba_texture, buffers.tmp_frequency_texture, tile_info_merge, textures.count, mode: ft_mode)
    // Convert back to 2x2 pixel structure
    // NOTE: Re-using `aligned_texture` since it is the correct size, it saves us from allocating another texture
    convert_bayer(buffers.aligned_rgba_texture, save_in: buffers.aligned_texture)
    // Add the cropped region to the final texture
    add_crop(buffers.aligned_texture, to: final_texture, x_offset: pad.left-crop_merge.x, y_offset: pad.top-crop_merge.y)
}


// ===========================================================================================================
// Helper functions
// ===========================================================================================================

func get_threads_per_thread_group(_ state: MTLComputePipelineState, _ threads_per_grid: MTLSize) -> MTLSize {
    var available_threads = state.maxTotalThreadsPerThreadgroup
    if threads_per_grid.depth > available_threads {
        return MTLSize(width: 1, height: 1, depth: available_threads)
    } else {
        available_threads /= threads_per_grid.depth
        if threads_per_grid.height > available_threads {
            return MTLSize(width: 1, height: available_threads, depth: threads_per_grid.depth)
        } else {
            available_threads /= threads_per_grid.height
            return MTLSize(width: available_threads, height: threads_per_grid.height, depth: threads_per_grid.depth)
        }
    }
}


func fill_with_zeros(_ texture: MTLTexture) {
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = fill_with_zeros_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: texture.width, height: texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(texture, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func texture_uint16_to_float(_ in_texture: MTLTexture) -> MTLTexture {

    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: in_texture.width, height: in_texture.height, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let out_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = texture_uint16_to_float_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return out_texture
}


func convert_uint16(_ in_texture: MTLTexture) -> MTLTexture {
    
    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Uint, width: in_texture.width, height: in_texture.height, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let out_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = convert_uint16_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: out_texture.width, height: out_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()

    return out_texture
}


func upsample(_ input_texture: MTLTexture, width: Int, height: Int, mode: String) -> MTLTexture {
    
    // convert args
    let scale_x = Double(width) / Double(input_texture.width)
    let scale_y = Double(height) / Double(input_texture.height)
    
    // create output texture
    let output_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: input_texture.pixelFormat, width: width, height: height, mipmapped: false)
    output_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let output_texture = device.makeTexture(descriptor: output_texture_descriptor)!
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = (mode == "bilinear" ? upsample_bilinear_float_state : upsample_nearest_int_state)
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: output_texture.width, height: output_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(input_texture, index: 0)
    command_encoder.setTexture(output_texture, index: 1)
    command_encoder.setBytes([Float32(scale_x)], length: MemoryLayout<Float32>.stride, index: 0)
    command_encoder.setBytes([Float32(scale_y)], length: MemoryLayout<Float32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return output_texture
}


func avg_pool(_ input_texture: MTLTexture, save_in out_texture: MTLTexture, _ scale: Int) {

    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = avg_pool_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: out_texture.width, height: out_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(input_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(scale)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func blur_mosaic_texture(_ in_texture: MTLTexture, _ kernel_size: Int, _ mosaic_pattern_width: Int) -> MTLTexture {
    
    // create a temp texture blurred along x-axis only and the output texture, blurred along both x- and y-axis
    let blur_x = texture_like(in_texture)
    let blur_xy = texture_like(in_texture)
    
    let kernel_size_mapped = (kernel_size == 16) ? 16 : max(0, min(8, kernel_size))
    
    // blur the texture along the x-axis
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = blur_mosaic_x_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(blur_x, index: 1)
    command_encoder.setBytes([Int32(kernel_size_mapped)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(mosaic_pattern_width)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)

    // blur the texture along the y-axis
    let state2 = blur_mosaic_y_state
    command_encoder.setComputePipelineState(state2)
    command_encoder.setTexture(blur_x, index: 0)
    command_encoder.setTexture(blur_xy, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return blur_xy
}


func extend_texture(_ in_texture: MTLTexture, onto out_texture: MTLTexture, _ pad_left: Int, _ pad_top: Int) {
            
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = extend_texture_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(pad_left)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(pad_top)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func crop_texture(_ in_texture: MTLTexture, _ pad_left: Int, _ pad_right: Int, _ pad_top: Int, _ pad_bottom: Int) -> MTLTexture {
    
    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: in_texture.pixelFormat, width: in_texture.width-pad_left-pad_right, height: in_texture.height-pad_top-pad_bottom, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let out_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = crop_texture_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: out_texture.width, height: out_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(pad_left)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(pad_top)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()

    return out_texture
}


func add_texture(_ in_texture: MTLTexture, to out_texture: MTLTexture, _ n_textures: Int) {
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = add_texture_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(n_textures)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}

func add_crop(_ in_texture: MTLTexture, to out_texture: MTLTexture, x_offset: Int, y_offset: Int) {
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = add_crop_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: out_texture.width, height: out_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(x_offset)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(y_offset)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func texture_like(_ input_texture: MTLTexture) -> MTLTexture {
    let output_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: input_texture.pixelFormat, width: input_texture.width, height: input_texture.height, mipmapped: false)
    output_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let output_texture = device.makeTexture(descriptor: output_texture_descriptor)!
    return output_texture
}

func copy_texture(_ in_texture: MTLTexture, onto out_texture: MTLTexture) {
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = copy_texture_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func copy_texture(_ in_texture: MTLTexture) -> MTLTexture {
    
    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: in_texture.pixelFormat, width: in_texture.width, height: in_texture.height, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let out_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    copy_texture(in_texture, onto: out_texture)
    
    return out_texture
}


func convert_rgba(_ in_texture: MTLTexture, save_in out_texture: MTLTexture, _ crop_merge_x: Int, _ crop_merge_y: Int){
        
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = convert_rgba_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: out_texture.width, height: out_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.setBytes([Int32(crop_merge_x)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(crop_merge_y)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func convert_bayer(_ in_texture: MTLTexture, save_in out_texture: MTLTexture) {
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = convert_bayer_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: in_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(out_texture, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func texture_mean(_ in_texture: MTLTexture, _ pixelformat: String) -> MTLBuffer {
    
    // If the parameter pixelformat has the value "rgba", the texture mean is calculated for the four color channels independently. An input texture in pixelformat "rgba" is expected for that purpose. In all other cases, a single mean value of the texture is calculated.
    
    // create a 1d texture that will contain the averages of the input texture along the x-axis
    let texture_descriptor = MTLTextureDescriptor()
    texture_descriptor.textureType = .type1D
    texture_descriptor.pixelFormat = in_texture.pixelFormat
    texture_descriptor.width = in_texture.width
    texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let avg_y = device.makeTexture(descriptor: texture_descriptor)!
    
    // average the input texture along the y-axis
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = (pixelformat == "rgba" ? average_y_rgba_state : average_y_state)
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: in_texture.width, height: 1, depth: 1)
    let max_threads_per_thread_group = state.maxTotalThreadsPerThreadgroup
    let threads_per_thread_group = MTLSize(width: max_threads_per_thread_group, height: 1, depth: 1)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(avg_y, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    
    // average the generated 1d texture along the x-axis
    let state2 = (pixelformat == "rgba" ? average_x_rgba_state : average_x_state)
    command_encoder.setComputePipelineState(state2)
    let avg_buffer = device.makeBuffer(length: (pixelformat=="rgba" ? 4 : 1)*MemoryLayout<Float32>.size, options: .storageModeShared)!
    command_encoder.setTexture(avg_y, index: 0)
    command_encoder.setBuffer(avg_buffer, offset: 0, index: 0)
    command_encoder.setBytes([Int32(in_texture.width)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    // return the average of all pixels in the input array
    return avg_buffer
}


/**
 * Helper function to create buffers needed for generating the pyramids.
 */
func generate_pyramid_buffers(with_metadata_from ref_texture: MTLTexture, with_pyramid_info pyramid_info: PyramidInfo, pad_align: DimensionalPadValues, tile_size: Int) -> (Array<MTLTexture>, Array<MTLTexture>) {
    var _ref_pyramid:  Array<MTLTexture> = []
    var _comp_pyramid: Array<MTLTexture> = []
    
    var width  = ref_texture.width  + 2*pad_align.x + tile_size
    var height = ref_texture.height + 2*pad_align.y + tile_size
        for downscale_factor in pyramid_info.downscale_factors {
            width  /= downscale_factor
            height /= downscale_factor
        let ref_pyramid_texture_descriptor_i = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: ref_texture.pixelFormat,
                                                                                        width: width,
                                                                                        height: height,
                                                                                        mipmapped: false)
        ref_pyramid_texture_descriptor_i.usage = [.shaderRead, .shaderWrite]
        let ref_pyramid_texture_i = device.makeTexture(descriptor: ref_pyramid_texture_descriptor_i)!
        
        let comp_pyramid_texture_descriptor_i = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: ref_texture.pixelFormat,
                                                                                         width: width,
                                                                                         height: height,
                                                                                         mipmapped: false)
        comp_pyramid_texture_descriptor_i.usage = [.shaderRead, .shaderWrite]
        let cmop_pyramid_texture_i = device.makeTexture(descriptor: comp_pyramid_texture_descriptor_i)!
        
        _ref_pyramid.append(ref_pyramid_texture_i)
        _comp_pyramid.append(cmop_pyramid_texture_i)
    }
    
    return (_ref_pyramid, _comp_pyramid)
}

func create_downscaling_pyramid_info(ref_texture: MTLTexture, mosaic_pattern_width: Int, tile_size: Int, search_distance: Int) -> PyramidInfo {
    // set alignment params
    let min_image_dim = min(ref_texture.width, ref_texture.height)
    var downscale_factor_array = [mosaic_pattern_width]
    var search_dist_array = [2]
    var tile_size_array = [tile_size]
    var res = min_image_dim / downscale_factor_array[0]
    var div = mosaic_pattern_width
    while (res > search_distance) {
        downscale_factor_array.append(2)
        search_dist_array.append(2)
        tile_size_array.append(max(tile_size_array.last!/2, 8))
        res /= 2
        div *= 2
    }
    
    return PyramidInfo(tile_sizes: tile_size_array,
                       search_dist: search_dist_array,
                       downscale_factors: downscale_factor_array,
                       tile_factor: div*Int(tile_size_array.last!))
}

// ===========================================================================================================
// Functions specific to image alignment
// ===========================================================================================================

func align_texture(_ ref_pyramid: [MTLTexture], _ comp_pyramid: [MTLTexture], _ comp_texture: MTLTexture, save_in aligned_texture: MTLTexture, _ pyramid_info: PyramidInfo) {
        
    // initialize tile alignments
    let alignment_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Sint, width: 1, height: 1, mipmapped: false)
    alignment_descriptor.usage = [.shaderRead, .shaderWrite]
    var prev_alignment = device.makeTexture(descriptor: alignment_descriptor)!
    var current_alignment = device.makeTexture(descriptor: alignment_descriptor)!
    var tile_info = TileInfo(tile_size: 0, tile_size_merge: 0, search_dist: 0, n_tiles_x: 0, n_tiles_y: 0, n_pos_1d: 0, n_pos_2d: 0)
    
    // build comparison pyramid
    build_pyramid(for_texture: comp_texture, save_in: comp_pyramid, pyramid_info.downscale_factors)
    
    // align tiles
    for i in (0 ... pyramid_info.downscale_factors.count-1).reversed() {
        
        // load layer params
        let tile_size = pyramid_info.tile_sizes[i]
        let search_dist = pyramid_info.search_dist[i]
        let ref_layer = ref_pyramid[i]
        let comp_layer = comp_pyramid[i]
        
        // calculate the number of tiles
        let n_tiles_x = ref_layer.width / (tile_size / 2) - 1
        let n_tiles_y = ref_layer.height / (tile_size / 2) - 1
        let n_pos_1d = 2*search_dist + 1
        let n_pos_2d = n_pos_1d * n_pos_1d
        
        // store tile info in a single dict
        tile_info = TileInfo(tile_size: tile_size, tile_size_merge: 0, search_dist: search_dist, n_tiles_x: n_tiles_x, n_tiles_y: n_tiles_y, n_pos_1d: n_pos_1d, n_pos_2d: n_pos_2d)
        
        // resize previous alignment
        // - 'downscale_factor' has to be loaded from the *previous* layer since that is the layer that generated the current layer
        var downscale_factor: Int
        if (i < pyramid_info.downscale_factors.count-1){
            downscale_factor = pyramid_info.downscale_factors[i+1]
        } else {
            downscale_factor = 0
        }
        
        // upsample alignment vectors by a factor of 2
        prev_alignment = upsample(current_alignment, width: n_tiles_x, height: n_tiles_y, mode: "nearest")
        
        // compare three alignment vector candidates, which improves alignment at borders of moving object
        // see https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf for more details
        prev_alignment = correct_upsampling_error(ref_layer, comp_layer, prev_alignment, downscale_factor, tile_info)
          
        // compute tile differences
        let tile_diff = compute_tile_diff(ref_layer, comp_layer, prev_alignment, downscale_factor, tile_info)
      
        current_alignment = texture_like(prev_alignment)
        
        // compute tile alignment based on tile differences
        compute_tile_alignment(tile_diff, prev_alignment, current_alignment, downscale_factor, tile_info)
    }
    
    // warp the aligned layer
    tile_info.tile_size *= pyramid_info.downscale_factors[0]
    warp_texture(comp_texture, current_alignment, save_in: aligned_texture, tile_info, pyramid_info.downscale_factors[0])
}


func build_pyramid(for_texture input_texture: MTLTexture, save_in pyramid: Array<MTLTexture>, _ downscale_factor_list: Array<Int>){
    
    // iteratively resize the current layer in the pyramid
    for (i, downscale_factor) in downscale_factor_list.enumerated() {
        if i == 0 {
            avg_pool(input_texture, save_in: pyramid[i], downscale_factor)
        } else {
            avg_pool(blur_mosaic_texture(pyramid[i-1], 2, 1), save_in: pyramid[i], downscale_factor)
        }
    }
}


func compute_tile_diff(_ ref_layer: MTLTexture, _ comp_layer: MTLTexture, _ prev_alignment: MTLTexture, _ downscale_factor: Int, _ tile_info: TileInfo) -> MTLTexture {
    
    // create a 'tile difference' texture
    let texture_descriptor = MTLTextureDescriptor()
    texture_descriptor.textureType = .type3D
    texture_descriptor.pixelFormat = .r32Float
    texture_descriptor.width = tile_info.n_tiles_x
    texture_descriptor.height = tile_info.n_tiles_y
    texture_descriptor.depth = tile_info.n_pos_2d
    texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let tile_diff = device.makeTexture(descriptor: texture_descriptor)!
    
    // compute tile differences
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    // either use generic function or highly-optimized function for testing a +/- 2 displacement in both image directions (in total 25 combinations)
    let state = (tile_info.n_pos_2d==25 ? compute_tile_differences25_state : compute_tile_differences_state)
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: (tile_info.n_pos_2d==25 ? 1 : tile_info.n_pos_2d))
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(ref_layer, index: 0)
    command_encoder.setTexture(comp_layer, index: 1)
    command_encoder.setTexture(prev_alignment, index: 2)
    command_encoder.setTexture(tile_diff, index: 3)
    command_encoder.setBytes([Int32(downscale_factor)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(tile_info.tile_size)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.setBytes([Int32(tile_info.search_dist)], length: MemoryLayout<Int32>.stride, index: 2)
    command_encoder.setBytes([Int32(tile_info.n_tiles_x)], length: MemoryLayout<Int32>.stride, index: 3)
    command_encoder.setBytes([Int32(tile_info.n_tiles_y)], length: MemoryLayout<Int32>.stride, index: 4)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return tile_diff
}


func compute_tile_alignment(_ tile_diff: MTLTexture, _ prev_alignment: MTLTexture, _ current_alignment: MTLTexture, _ downscale_factor: Int, _ tile_info: TileInfo) {
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = compute_tile_alignments_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(tile_diff, index: 0)
    command_encoder.setTexture(prev_alignment, index: 1)
    command_encoder.setTexture(current_alignment, index: 2)
    command_encoder.setBytes([Int32(downscale_factor)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(tile_info.search_dist)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func correct_upsampling_error(_ ref_layer: MTLTexture, _ comp_layer: MTLTexture, _ prev_alignment: MTLTexture, _ downscale_factor: Int, _ tile_info: TileInfo) -> MTLTexture {
    
    // create texture for corrected alignment
    let prev_alignment_corrected_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: prev_alignment.pixelFormat, width: prev_alignment.width, height: prev_alignment.height, mipmapped: false)
    prev_alignment_corrected_descriptor.usage = [.shaderRead, .shaderWrite]
    let prev_alignment_corrected = device.makeTexture(descriptor: prev_alignment_corrected_descriptor)!
        
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = correct_upsampling_error_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(ref_layer, index: 0)
    command_encoder.setTexture(comp_layer, index: 1)
    command_encoder.setTexture(prev_alignment, index: 2)
    command_encoder.setTexture(prev_alignment_corrected, index: 3)
    command_encoder.setBytes([Int32(downscale_factor)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(tile_info.tile_size)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.setBytes([Int32(tile_info.n_tiles_x)], length: MemoryLayout<Int32>.stride, index: 2)
    command_encoder.setBytes([Int32(tile_info.n_tiles_y)], length: MemoryLayout<Int32>.stride, index: 3)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return prev_alignment_corrected
}


func warp_texture(_ texture_to_warp: MTLTexture, _ alignment: MTLTexture, save_in warped_texture: MTLTexture, _ tile_info: TileInfo, _ downscale_factor: Int){
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = warp_texture_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: texture_to_warp.width, height: texture_to_warp.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(texture_to_warp, index: 0)
    command_encoder.setTexture(warped_texture, index: 1)
    command_encoder.setTexture(alignment, index: 2)
    command_encoder.setBytes([Int32(downscale_factor)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(tile_info.tile_size)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.setBytes([Int32(tile_info.n_tiles_x)], length: MemoryLayout<Int32>.stride, index: 2)
    command_encoder.setBytes([Int32(tile_info.n_tiles_y)], length: MemoryLayout<Int32>.stride, index: 3)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


// ===========================================================================================================
// Functions specific to merging in the spatial domain
// ===========================================================================================================

func add_textures_weighted(_ texture1: MTLTexture, _ texture2: MTLTexture, _ weight_texture: MTLTexture) -> MTLTexture {
    
    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture1.pixelFormat, width: texture1.width, height: texture1.height, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let out_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    // add textures
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = add_textures_weighted_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: texture1.width, height: texture1.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(texture1, index: 0)
    command_encoder.setTexture(texture2, index: 1)
    command_encoder.setTexture(weight_texture, index: 2)
    command_encoder.setTexture(out_texture, index: 3)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return out_texture
}


func color_difference(_ texture1: MTLTexture, _ texture2: MTLTexture, _ mosaic_pattern_width: Int) -> MTLTexture {
    
    let out_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: texture1.pixelFormat, width: texture1.width/mosaic_pattern_width, height: texture1.height/mosaic_pattern_width, mipmapped: false)
    out_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let output_texture = device.makeTexture(descriptor: out_texture_descriptor)!
    
    // compute pixel pairwise differences
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = color_difference_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: texture1.width/2, height: texture1.height/2, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(texture1, index: 0)
    command_encoder.setTexture(texture2, index: 1)
    command_encoder.setTexture(output_texture, index: 2)
    command_encoder.setBytes([Int32(mosaic_pattern_width)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return output_texture
}


func estimate_color_noise(_ texture: MTLTexture, _ texture_blurred: MTLTexture, _ mosaic_pattern_width: Int) -> MTLBuffer {
    
    // compute the color difference of each mosaic superpixel between the original and the blurred texture
    let texture_diff = color_difference(texture, texture_blurred, mosaic_pattern_width)
    
    // compute the average of the difference between the original and the blurred texture
    let mean_diff = texture_mean(texture_diff, "r")
    
    return mean_diff
}


func robust_merge(_ ref_texture: MTLTexture, _ ref_texture_blurred: MTLTexture, _ comp_texture: MTLTexture, _ kernel_size: Int, _ robustness: Double, _ noise_sd: MTLBuffer, _ mosaic_pattern_width: Int) -> MTLTexture {
    
    // blur comparison texture
    let comp_texture_blurred = blur_mosaic_texture(comp_texture, kernel_size, mosaic_pattern_width)
    
    // compute the color difference of each superpixel between the blurred reference and the comparison textures
    let texture_diff = color_difference(ref_texture_blurred, comp_texture_blurred, mosaic_pattern_width)
    
    // create a weight texture
    let weight_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: texture_diff.width, height: texture_diff.height, mipmapped: false)
    weight_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let weight_texture = device.makeTexture(descriptor: weight_texture_descriptor)!
    
    // compute merge weight
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = compute_merge_weight_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: texture_diff.width, height: texture_diff.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(texture_diff, index: 0)
    command_encoder.setTexture(weight_texture, index: 1)
    command_encoder.setBuffer(noise_sd, offset: 0, index: 0)
    command_encoder.setBytes([Float32(robustness)], length: MemoryLayout<Float32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    // upsample merge weight to full image resolution
    let weight_texture_upsampled = upsample(weight_texture, width: ref_texture.width, height: ref_texture.height, mode: "bilinear")
    
    // average the input textures based on the weight
    let output_texture = add_textures_weighted(ref_texture, comp_texture, weight_texture_upsampled)
    
    return output_texture
}


// ===========================================================================================================
// Functions specific to merging in the frequency domain
// ===========================================================================================================

/**
 * Helper function that takes care of creating all of the temporary buffers used by the frequency-merge approach.
 * These buffers are created once and then re-used through the frequency merging in order to minimize memory allocations/deallocation.
 */
func create_frequency_buffers(ref_texture: MTLTexture, mosaic_pattern_width: Int, search_distance: Int, tile_size: Int, tile_size_merge: Int, pyramid_info: PyramidInfo) -> (FrequencyMergeTextureBuffers, DimensionalPadValues, DimensionalPadValues, TileInfo) {
    // set original texture size
    let texture_width_orig = ref_texture.width
    let texture_height_orig = ref_texture.height
     
    // calculate padding for extension of the image frame with zeros in such a way that the original image resolution is a multiple of all resolution levels used during alignment and nearest neighbor upsampling of motion vector fields is sufficient
    var _pad_align_x = Int(ceil(Float(texture_width_orig+tile_size_merge)/Float(pyramid_info.tile_factor)))
    _pad_align_x = (_pad_align_x*Int(pyramid_info.tile_factor) - texture_width_orig - tile_size_merge)/2
    
    var _pad_align_y = Int(ceil(Float(texture_height_orig+tile_size_merge)/Float(pyramid_info.tile_factor)))
    _pad_align_y = (_pad_align_y*Int(pyramid_info.tile_factor) - texture_height_orig - tile_size_merge)/2
    
    let pad_align = DimensionalPadValues(x: _pad_align_x, y: _pad_align_y)
  
    // calculate padding for the merging in the frequency domain, which can be applied to the actual image frame + a smaller margin compared to the alignment
    let crop_merge = DimensionalPadValues(x: Int(floor(Float(pad_align.x)/Float(2*tile_size_merge))) * 2 * tile_size_merge,
                                          y: Int(floor(Float(pad_align.y)/Float(2*tile_size_merge))) * 2 * tile_size_merge)
    
    let pad_merge = DimensionalPadValues(x: pad_align.x - crop_merge.x,
                                         y: pad_align.y - crop_merge.y)
    
    // set tile information needed for the merging
    let tile_info_merge = TileInfo(
        tile_size: tile_size,
        tile_size_merge: tile_size_merge,
        search_dist: 0,
        n_tiles_x: (texture_width_orig  + tile_size_merge + 2*pad_merge.x) / (2*tile_size_merge),
        n_tiles_y: (texture_height_orig + tile_size_merge + 2*pad_merge.y) / (2*tile_size_merge),
        n_pos_1d: 0,
        n_pos_2d: 0
    )
    
    let ref_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float,
                                                                          width:  texture_width_orig + 2*pad_align.x + tile_size_merge,
                                                                          height: texture_height_orig + 2*pad_align.y + tile_size_merge,
                                                                          mipmapped: false)
    ref_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let _ref_texture = device.makeTexture(descriptor: ref_texture_descriptor)!
    fill_with_zeros(_ref_texture)

    let _comp_texture = texture_like(_ref_texture)
    fill_with_zeros(_comp_texture)
    
    let _aligned_texture = texture_like(_ref_texture)
    fill_with_zeros(_aligned_texture)
    
    let final_frequency_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                                      width:  (texture_width_orig  + tile_size_merge + 2*pad_merge.x),
                                                                                      height: (texture_height_orig + tile_size_merge + 2*pad_merge.y)/2,
                                                                                      mipmapped: false)
    final_frequency_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let _final_frequency_texture = device.makeTexture(descriptor: final_frequency_texture_descriptor)!
    fill_with_zeros(_final_frequency_texture)
    
    let _tmp_texture_ft = texture_like(_final_frequency_texture)
    let _ref_texture_ft = texture_like(_final_frequency_texture)
    let _aligned_texture_ft = texture_like(_final_frequency_texture)
       
    let aligned_texture_rgba_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                                   width:  (_ref_texture.width  - 2*crop_merge.x)/2,
                                                                                   height: (_ref_texture.height - 2*crop_merge.y)/2,
                                                                                   mipmapped: false)
    aligned_texture_rgba_descriptor.usage = [.shaderRead, .shaderWrite]
    let _aligned_texture_rgba = device.makeTexture(descriptor: aligned_texture_rgba_descriptor)!
    
    let _ref_texture_rgba = texture_like(_aligned_texture_rgba)
    
    let rms_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                          width: tile_info_merge.n_tiles_x,
                                                                          height: tile_info_merge.n_tiles_y,
                                                                          mipmapped: false)
    rms_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let _rms_rgba_texture = device.makeTexture(descriptor: rms_texture_descriptor)!
    
    // generate texture to accumulate the total mismatch
    let _total_mismatch_rgba_texture = texture_like(_rms_rgba_texture)
    fill_with_zeros(_total_mismatch_rgba_texture)

    let (_ref_pyramid, _comp_pyramid) = generate_pyramid_buffers(with_metadata_from: ref_texture, with_pyramid_info: pyramid_info, pad_align: pad_align, tile_size: tile_size_merge)
    
    let buffers = FrequencyMergeTextureBuffers(
        // Frequency-textures
        final_frequency_texture:    _final_frequency_texture,
        ref_frequency_texture:      _ref_texture_ft,
        aligned_frequency_texture:  _aligned_texture_ft,
        tmp_frequency_texture:      _tmp_texture_ft,
        // RGBA-texture
        ref_rgba_texture:               _ref_texture_rgba,
        aligned_rgba_texture:           _aligned_texture_rgba,
        // R-textures
        ref_texture:        _ref_texture,
        comp_texture:       _comp_texture,
        aligned_texture:    _aligned_texture,
        // Pyramid-textures
        ref_pyramid:  _ref_pyramid,
        comp_pyramid: _comp_pyramid
    )
    
    return (buffers, crop_merge, pad_align, tile_info_merge)
}


func calculate_rms_rgba(_ in_texture: MTLTexture, _ tile_info: TileInfo) -> MTLTexture {
    
    let rms_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, mipmapped: false)
    rms_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let rms_texture = device.makeTexture(descriptor: rms_texture_descriptor)!
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = calculate_rms_rgba_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: rms_texture.width, height: rms_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(rms_texture, index: 1)
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return rms_texture
}


func forward_ft(_ in_texture: MTLTexture, save_in out_texture_ft: MTLTexture, _ tmp_texture_ft: MTLTexture, _ tile_info: TileInfo, mode: String) {
  
    var cosine_factor = 0.500    
    // adapt cosine factor dependent on tile size
    if (Int32(tile_info.tile_size_merge) == 8) {cosine_factor = 0.530}
    else if (Int32(tile_info.tile_size_merge) == 16) {cosine_factor = 0.505}
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    // either use discrete Fourier transform or highly-optimized fast Fourier transform
    let state = (mode == "DFT" ? forward_dft_state : forward_fft_state)
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture, index: 0)
    command_encoder.setTexture(tmp_texture_ft, index: 1)
    command_encoder.setTexture(out_texture_ft, index: 2)
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Float32(cosine_factor)], length: MemoryLayout<Float32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func backward_ft(_ in_texture_ft: MTLTexture, save_in out_texture: MTLTexture, _ tmp_texture_ft: MTLTexture, _ tile_info: TileInfo, _ n_textures: Int, mode: String) {
    
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    // either use discrete Fourier transform or highly-optimized fast Fourier transform
    let state = (mode == "DFT" ? backward_dft_state : backward_fft_state)
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(in_texture_ft, index: 0)
    command_encoder.setTexture(tmp_texture_ft, index: 1)
    command_encoder.setTexture(out_texture, index: 2)
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.setBytes([Int32(n_textures)], length: MemoryLayout<Int32>.stride, index: 1)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func merge_frequency_domain(_ ref_texture_ft: MTLTexture, _ aligned_texture_ft: MTLTexture, _ out_texture_ft: MTLTexture, _ rms_texture: MTLTexture, _ mismatch_texture: MTLTexture, _ robustness_norm: Double, _ read_noise: Double, _ max_motion_norm: Double, _ tile_info: TileInfo) {
        
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = merge_frequency_domain_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(ref_texture_ft, index: 0)
    command_encoder.setTexture(aligned_texture_ft, index: 1)
    command_encoder.setTexture(out_texture_ft, index: 2)
    command_encoder.setTexture(rms_texture, index: 3)
    command_encoder.setTexture(mismatch_texture, index: 4)
    command_encoder.setBytes([Float32(robustness_norm)], length: MemoryLayout<Float32>.stride, index: 0)
    command_encoder.setBytes([Float32(read_noise)], length: MemoryLayout<Float32>.stride, index: 1)
    command_encoder.setBytes([Float32(max_motion_norm)], length: MemoryLayout<Float32>.stride, index: 2)    
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 3)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func deconvolute_frequency_domain(_ final_texture_ft: MTLTexture, _ total_mismatch_texture: MTLTexture, _ tile_info: TileInfo) {
        
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = deconvolute_frequency_domain_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(final_texture_ft, index: 0)
    command_encoder.setTexture(total_mismatch_texture, index: 1)
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


func calculate_mismatch_rgba(_ aligned_texture: MTLTexture, _ ref_texture: MTLTexture, _ rms_texture: MTLTexture, _ tile_info: TileInfo) -> MTLTexture {
  
    // create mismatch texture
    let mismatch_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, mipmapped: false)
    mismatch_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let mismatch_texture = device.makeTexture(descriptor: mismatch_texture_descriptor)!
               
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = calculate_mismatch_rgba_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: tile_info.n_tiles_x, height: tile_info.n_tiles_y, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(aligned_texture, index: 0)
    command_encoder.setTexture(ref_texture, index: 1)
    command_encoder.setTexture(rms_texture, index: 2)
    command_encoder.setTexture(mismatch_texture, index: 3)
    command_encoder.setBytes([Int32(tile_info.tile_size_merge)], length: MemoryLayout<Int32>.stride, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
    
    return mismatch_texture
}


func normalize_mismatch(_ mismatch_texture: MTLTexture, _ mean_mismatch_buffer: MTLBuffer) {
   
    let command_buffer = command_queue.makeCommandBuffer()!
    let command_encoder = command_buffer.makeComputeCommandEncoder()!
    let state = normalize_mismatch_state
    command_encoder.setComputePipelineState(state)
    let threads_per_grid = MTLSize(width: mismatch_texture.width, height: mismatch_texture.height, depth: 1)
    let threads_per_thread_group = get_threads_per_thread_group(state, threads_per_grid)
    command_encoder.setTexture(mismatch_texture, index: 0)
    command_encoder.setBuffer(mean_mismatch_buffer, offset: 0, index: 0)
    command_encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group)
    command_encoder.endEncoding()
    command_buffer.commit()
}


/**
 * Find the hotpixels in the images and correct them based on the average of adjacent pixels.
 * Performed in a two-passes in order to avoid needing to copy any of the textures.
 * Pass 1 identifies the hot pixels and stores their locations.
 * Pass 2 corrects the hot pixels based on neighbouring values.
 *
 * The textures are assumed to be floats and not unsigned integers. -1 will be used to identify non-hotpixels.
 */
func correct_hotpixels(_ textures: [MTLTexture]) {
    
    // generate simple average of all textures
    let average_texture_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: textures[0].width, height: textures[0].height, mipmapped: false)
    average_texture_descriptor.usage = [.shaderRead, .shaderWrite]
    let average_texture = device.makeTexture(descriptor: average_texture_descriptor)!
    fill_with_zeros(average_texture)
    for texture in textures {
        add_texture(texture, to: average_texture, textures.count)
    }
    
    // Texture to stores whethere a hot pixel was identifies
    // A hotpixel will have the mean of the 4 neighbours pixels stored
    // A non-notpixel will have -1.0 stored
    let hot_pixel_identified_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: textures[0].width, height: textures[0].height, mipmapped: false)
    hot_pixel_identified_descriptor.usage = [.shaderRead, .shaderWrite]
    let hot_pixel_identified = device.makeTexture(descriptor: hot_pixel_identified_descriptor)!
    fill_with_zeros(hot_pixel_identified)
    
    // calculate mean value specific for each color channel
    let mean_texture_rgba_descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: average_texture.width/2, height: average_texture.height/2, mipmapped: false)
    mean_texture_rgba_descriptor.usage = [.shaderRead, .shaderWrite]
    let mean_texture_rgba = device.makeTexture(descriptor: mean_texture_rgba_descriptor)!
    convert_rgba(average_texture, save_in: mean_texture_rgba, 0, 0)
    let mean_texture_buffer = texture_mean(mean_texture_rgba,"rgba")
    
    
    // iterate over all images and correct hot pixels in each texture
    for texture in textures {
        // Pass 1: Identify hot pixels
        let command_buffer_1 = command_queue.makeCommandBuffer()!
        let command_encoder_1 = command_buffer_1.makeComputeCommandEncoder()!
        command_encoder_1.setComputePipelineState(identify_hotpixels_state)
        let threads_per_grid = MTLSize(width: average_texture.width-4, height: average_texture.height-4, depth: 1)
        let threads_per_thread_group_1 = get_threads_per_thread_group(identify_hotpixels_state, threads_per_grid)
        
        command_encoder_1.setTexture(average_texture, index: 0)
        command_encoder_1.setTexture(texture, index: 1)
        command_encoder_1.setTexture(hot_pixel_identified, index: 2)
        command_encoder_1.setBuffer(mean_texture_buffer, offset: 0, index: 0)
        command_encoder_1.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group_1)
        command_encoder_1.endEncoding()
        command_buffer_1.commit()
        
        // Pass 2: Fix hot pixels
        let command_buffer_2 = command_queue.makeCommandBuffer()!
        let command_encoder_2 = command_buffer_2.makeComputeCommandEncoder()!
        command_encoder_2.setComputePipelineState(correct_hotpixels_state)
        let threads_per_thread_group_2 = get_threads_per_thread_group(correct_hotpixels_state, threads_per_grid)
        
        command_encoder_2.setTexture(average_texture, index: 0)
        command_encoder_2.setTexture(hot_pixel_identified, index: 1)
        command_encoder_2.setTexture(texture, index: 2)
        command_encoder_2.setBuffer(mean_texture_buffer, offset: 0, index: 0)
        command_encoder_2.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_thread_group_2)
        command_encoder_2.endEncoding()
        command_buffer_2.commit()
    }
}

