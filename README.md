# C++ implementation of PRNet
This project contains:
1. Face reconstruct(Face swap) from only a single image.
2. Face keypoints(68) in 3D model.
3. Face pose estimate in 3D model.

# Platfrom and required
    1. iPhone SE, arm64
    2. OpenCV 3.4.0
    3. NCNN
    
# Outline
1. Mtcnn --> resize to 256x256 --> PRNet --> UV,Z --> Render 86906 triangles in 3D --> SeamlessClone

                                 pick 68 points for sparse alignment
                                                   ^
                                                   |
2. Mtcnn --> resize to 256x256 --> PRNet --> 43867 vertices --> SVD to estimate pose

                                                   

# Application
1. Face Swap, change the target face image(@"ref.jpg") to your own.

2. Face 3D pose estimate and face keypoints in 3D.

# About canonical config
**There's 4 config files in this project.
