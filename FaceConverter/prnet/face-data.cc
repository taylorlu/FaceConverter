#include "face-data.h"

#include <iostream>
#include <fstream>
#include <sstream>

namespace {

std::string JoinPath(const std::string &dir, const std::string &filename) {
  if (dir.empty()) {
    return filename;
  } else {
    // check '/'
    char lastChar = *dir.rbegin();
    if (lastChar != '/') {
      return dir + std::string("/") + filename;
    } else {
      return dir + filename;
    }
  }
}

} // namespace 

bool LoadFaceData(const std::string &datapath, FaceData *face_data)
{
  face_data->face_indices.clear();
  face_data->triangles.clear();

  // Load face index data.
  {
    std::string face_idx_filename = JoinPath(datapath, "face_ind.txt");
    std::ifstream ifs(face_idx_filename);
    if (!ifs) {
      std::cerr << "File not found or failed to open : " << face_idx_filename << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
      // `face_ind.txt` stores integer data in scientific fp value.
      // So first read as float, then cast to int.
      uint32_t idx = static_cast<uint32_t>(std::stof(line));
      face_data->face_indices.push_back(idx);
      //std::cout << idx << std::endl;
    } 

  }

  // Load triangle data.
  {
    std::string triangles_filename = JoinPath(datapath, "triangles.txt");
    std::ifstream ifs(triangles_filename);
    if (!ifs) {
      std::cerr << "File not found or failed to open : " << triangles_filename << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
      // `triangles.txt` stores integer data in scientific fp value.
      // So first read as float, then cast to int.
      std::stringstream ss(line);
      std::string xs, ys, zs;

      ss >> xs >> ys >> zs;
 
      uint32_t v0 = static_cast<uint32_t>(std::stof(xs));
      uint32_t v1 = static_cast<uint32_t>(std::stof(ys));
      uint32_t v2 = static_cast<uint32_t>(std::stof(zs));
      face_data->triangles.push_back(v0);
      face_data->triangles.push_back(v1);
      face_data->triangles.push_back(v2);
      //std::cout << v0 << ", " << v1 << ", " << v2 << std::endl;
    } 
  }

  // Loads corresponding uv(pixel) location for landmark points in an image.
  {
    std::string uv_filename = JoinPath(datapath, "uv_kpt_ind.txt");
    std::ifstream ifs(uv_filename);
    if (!ifs) {
      std::cerr << "File not found or failed to open : " << uv_filename << std::endl;
      return false;
    }

    std::string s;
    while (ifs >> s) {
      uint32_t val = static_cast<uint32_t>(std::stof(s));
      face_data->uv_kpt_indices.push_back(val);
      // std::cout << val << std::endl;
    }

    if (face_data->uv_kpt_indices.size() != (2 * 68)) {
      std::cerr << "Invalid number of UV values. Must be 2 * 68, but got " << face_data->uv_kpt_indices.size() << std::endl;
      return false; 
    }
  }
    
    // Load canonical vertices
    {
        std::string triangles_filename = JoinPath(datapath, "canonical_vertices.txt");
        std::ifstream ifs(triangles_filename);
        if (!ifs) {
            std::cerr << "File not found or failed to open : " << triangles_filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(ifs, line)) {
            std::stringstream ss(line);
            std::string xs, ys, zs;
            
            ss >> xs >> ys >> zs;
            
            float x = std::stof(xs);
            float y = std::stof(ys);
            float z = std::stof(zs);
            
            face_data->canonical_vertices.push_back(std::array<float, 3>{x, y, z});
        }
        
        if (face_data->canonical_vertices.size() != 43867) {
            std::cerr << "Invalid number of canonical vertices. Must be 43867, but got " << face_data->canonical_vertices.size() << std::endl;
            return false;
        }
    }
  return true;
}


