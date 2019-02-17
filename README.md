# C++ implementation of PRNet
This project contains:
1. Face reconstruct(Face swap) from only a single image.
2. Face keypoints(68) in 3D model.
3. Face pose estimate in 3D model.

# Platfrom and requirements
    1. iPhone SE, arm64
    2. OpenCV 3.4.0
    3. NCNN
    
# Outline
1. Mtcnn --> resize to 256x256 --> PRNet --> UV,Z --> Render 86906 triangles in 3D --> SeamlessClone

2. Mtcnn --> resize to 256x256 --> PRNet --> 43867 vertices --> SVD to estimate pose
   Mtcnn --> resize to 256x256 --> PRNet --> 43867 vertices --> pick 68 points for sparse alignment
                                                   

# Application
1. Face Swap, change the target face image(@"ref.jpg") to your own.
2. Face 3D pose estimate and face keypoints in 3D.
<div align="center">
<img src="https://github.com/taylorlu/FaceConverter/tree/master/FaceConverter/pics/show-1.jpg" height="414" width="240" >
<img src="https://github.com/taylorlu/FaceConverter/tree/master/FaceConverter/pics/show-2.jpg" height="414" width="240" >
</div>

# About the configuration
**There's 4 config files in this project.

# Other details
1. About the Euler angles.

http://www.gregslabaugh.net/publications/euler.pdf

2. 3D points transformation matrix.

http://nghiaho.com/?page_id=671

3. Render 3D texture.
