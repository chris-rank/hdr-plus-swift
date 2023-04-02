import Foundation
import MetalKit


func optionally_convert_dir_to_urls(_ urls: [URL]) -> [URL] {
    // if the argument is a list of one directory, return the urls withing that directory
    if urls.count == 1 {
        let url = urls[0]
        if let url_is_dir = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory {
            if url_is_dir {
                let fm = FileManager.default
                if let updated_urls = try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [], options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]) {
                    return updated_urls
                }
            }
        }
    }
    return urls
}

func load_images(_ urls: [URL]) throws -> ([MTLTexture], Int) {
    
    var textures_dict: [Int: MTLTexture] = [:]
    let compute_group = DispatchGroup()
    let compute_queue = DispatchQueue.global() // this is a concurrent queue to do compute
    let access_queue = DispatchQueue(label: "") // this is a serial queue to read/save data thread-safely
    var mosaic_pattern_width: Int?

    for i in 0..<urls.count {
        compute_queue.async(group: compute_group) {
    
            // asynchronously load texture
            if let (texture, _mosaic_pattern_width) = try? image_url_to_texture(urls[i], device) {
        
                // thread-safely save the texture
                access_queue.sync {
                    textures_dict[i] = texture
                    mosaic_pattern_width = _mosaic_pattern_width
                }
            }
        }
    }
    
    // wait until all the images are loaded
    compute_group.wait()
    
    // convert dict to list
    var textures_list: [MTLTexture] = []
    for i in 0..<urls.count {
        
        // ensure thread-safety
        try access_queue.sync {
            
            // check whether the images have been loaded successfully
            if let texture = textures_dict[i] {
                textures_list.append(texture)
            } else {
                throw ImageIOError.load_error
            }
        }
    }
    
    // Verify that all textures are of the same size
    for i in 1..<textures_list.count {
        if (textures_list[0].width != textures_list[i].width)
            || (textures_list[0].height != textures_list[i].height) {
            throw AlignmentError.inconsistent_resolutions
        }
    }
    
    return (textures_list, mosaic_pattern_width!)
}

// https://stackoverflow.com/questions/26971240/how-do-i-run-a-terminal-command-in-a-swift-script-e-g-xcodebuild
@discardableResult // Add to suppress warnings when you don't want/need a result
func safeShell(_ command: String) throws -> String {
    let task = Process()
    let pipe = Pipe()
    
    task.standardOutput = pipe
    task.standardError = pipe
    task.arguments = ["-c", command]
    task.executableURL = URL(fileURLWithPath: "/bin/zsh") //<--updated
    task.standardInput = nil

    try task.run() //<--updated
    
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: data, encoding: .utf8)!
    
    return output
}

func convert_images_to_dng(_ in_urls: [URL], _ dng_converter_path: String, _ tmp_dir: String) throws -> [URL] {

    // create command string
    let executable_path = dng_converter_path + "/Contents/MacOS/Adobe DNG Converter"
    // changed args to convert raw files to compressed DNGs
    let args = "--args -c -p0 -d \"\(tmp_dir)\""
    var command = "\"\(executable_path)\" \(args)"
    for url in in_urls {
        command += " \"\(url.relativePath)\""
    }

    // call adobe dng converter
    let output = try safeShell(command)

    // return urls of the newly created dngs
    var out_urls: [URL] = []
    for url in in_urls {
        let fine_name = url.deletingPathExtension().lastPathComponent + ".dng"
        let out_path = tmp_dir + fine_name
        let out_url = URL(fileURLWithPath: out_path)
        out_urls.append(out_url)
        if !FileManager.default.fileExists(atPath: out_path) {
            throw AlignmentError.conversion_failed
        }
    }

    return out_urls
}
