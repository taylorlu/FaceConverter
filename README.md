# C++ implementation of PRNet on iOS
This project contains:
1. Face reconstruct(Face swap) from only a single image.
2. Face pose estimate and keypoints(68) in 3D model.

# Platfrom and requirements
    1. iPhone SE, arm64
    2. OpenCV 3.4.0
    3. NCNN
    
# Outline
1. Mtcnn --> resize to 256x256 --> PRNet --> UV,Z --> Render 86906 triangles in 3D --> SeamlessClone
2. Mtcnn --> resize to 256x256 --> PRNet --> 43867 vertices --> SVD to estimate pose</br>
   Mtcnn --> resize to 256x256 --> PRNet --> 43867 vertices --> pick 68 points for sparse alignment
<div align="center">
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/mesh.png" height="320" width="600">
</div>

# Application
1. Face Swap, change the target face image(@"ref.jpg") to your own.
2. Face 3D pose estimate and face keypoints in 3D.

<div align="center">
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-1.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-2.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-3.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-4.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-5.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-6.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-7.jpg" height="360" width="640" >
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/show-8.jpg" height="360" width="640" >
</div>

# About the configuration
There's 4 config files in this project.

    1. face_ind.txt
This file is the index of [0,65535], which is the region of the **WHITE** area of the image[256x256] below.
<div align="center">
<img src="https://github.com/taylorlu/FaceConverter/blob/master/FaceConverter/pics/uv_face_mask.png" height="256" width="256">
</div>

    2. uv_kpt_ind.txt
This file is the index of coordinate(x,y) for 68 keypoints refer to face_ind.txt

    3. triangles.txt
86906 triangles' vertices index refer to face_ind.txt

    4. canonical_vertices.txt
canonical model for pose estimate, there's coordinate (uv,z) of 43867 vertices

# Other details
1. About the Euler angles.

http://www.gregslabaugh.net/publications/euler.pdf

2. 3D points transformation matrix.

http://nghiaho.com/?page_id=671

3. Render 3D texture.
