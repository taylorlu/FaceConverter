//
//  ComViewController.m
//  VideoFace
//
//  Created by LuDong on 2018/7/2.
//  Copyright © 2018年 LuDong. All rights reserved.
//

#import "ComViewController.h"
#import <CoreGraphics/CGContext.h>


@interface ComViewController ()

@end

@implementation ComViewController

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

bool isInside(cv::Rect rect1, cv::Rect rect2) {
    return (rect1 == (rect1&rect2));
}

cv::Mat drawKeyPoint(const cv::Mat &img, vector<uint32_t> realPos) {

    std::set<uint32_t> end_list = {16, 21, 26, 41, 47, 30, 35, 67};
    if(realPos.size()==0) {
        return img;
    }
    cv::Mat show = img.clone();
    for (int i = 0; i < KPT_COUNT; i++) {
        cv::circle(show, cvPoint(realPos[i*2], realPos[i*2+1]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        auto search = end_list.find(i);
        if (search != end_list.end()) {
            continue;
        }
        cv::line(show, cvPoint(realPos[i*2], realPos[i*2+1]), cvPoint(realPos[(i+1)*2], realPos[(i+1)*2+1]), CV_RGB(255, 255, 0));
    }
    return show;
}
cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box) {
    
    cv::Mat show = img.clone();
    int num_box = (int)box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1, box[i].y2 - box[i].y1 + 1);
        
        for (int j = 0; j < 5; j = j + 1) {
            cv::circle(show, cvPoint(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    int i=0;
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(show, (*it), cv::Scalar(255, 127, 0), 2, 8, 0);
        cv::putText(show, box[i++].text, cvPoint((*it).x, (*it).y), cv::FONT_HERSHEY_COMPLEX, 0.8, CV_RGB(0, 255, 0),1);
    }
    return show;
}

-(MLMultiArray *)facePRNetCoreML:(cv::Mat) faceCrop {  //usr coreml.
    
    cv::resize(faceCrop, faceCrop, cv::Size(256, 256));
    vector<cv::Mat> xc;
    split(faceCrop, xc);
    
    int count = 0;
    for(int i=0; i<256*256; i++) {
        inputData[count++] = *(xc[2].data+i)/256.0;
    }
    for(int i=0; i<256*256; i++) {
        inputData[count++] = *(xc[1].data+i)/256.0;
    }
    for(int i=0; i<256*256; i++) {
        inputData[count++] = *(xc[0].data+i)/256.0;
    }
    
    MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:inputData shape:[NSArray arrayWithObjects:[NSNumber numberWithInt:3], [NSNumber numberWithInt:256], [NSNumber numberWithInt:256], nil] dataType:MLMultiArrayDataTypeDouble strides:[NSArray arrayWithObjects:[NSNumber numberWithInt:256*256], [NSNumber numberWithInt:256], [NSNumber numberWithInt:1], nil] deallocator:nil error:nil];
    
    prnetOutput *output = [irModel predictionFromPlaceholder__0:arr error:nil];
    MLMultiArray *multiArr = [output resfcn256__Conv2d_transpose_16__Sigmoid__0];
    
    return multiArr;
}

//-(cv::Mat)facePRNetCoreML:(cv::Mat) faceCrop {  //usr coreml.
//
//    cv::resize(faceCrop, faceCrop, cv::Size(256, 256));
//    vector<cv::Mat> xc;
//    split(faceCrop, xc);
//
//    int count = 0;
//    for(int i=0; i<256*256; i++) {
//        inputData[count++] = *(xc[0].data+i);
//    }
//    for(int i=0; i<256*256; i++) {
//        inputData[count++] = *(xc[1].data+i);
//    }
//    for(int i=0; i<256*256; i++) {
//        inputData[count++] = *(xc[2].data+i);
//    }
//
//    MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:inputData shape:[NSArray arrayWithObjects:[NSNumber numberWithInt:3], [NSNumber numberWithInt:256], [NSNumber numberWithInt:256], nil] dataType:MLMultiArrayDataTypeDouble strides:[NSArray arrayWithObjects:[NSNumber numberWithInt:256*256], [NSNumber numberWithInt:256], [NSNumber numberWithInt:1], nil] deallocator:nil error:nil];
//
//    prnetOutput *output = [irModel predictionFromPlaceholder__0:arr error:nil];
//    MLMultiArray *multiArr = [output resfcn256__Conv2d_transpose_16__Sigmoid__0];
//
//    cv::Mat posMat1;
//    posMat1.create(256,256, CV_64F);
//    cv::Mat posMat2;
//    posMat2.create(256,256, CV_64F);
//    cv::Mat posMat3;
//    posMat3.create(256,256, CV_64F);
//
//    int plannerSize = [[multiArr strides][0] intValue];
//    memcpy(posMat1.data, [multiArr dataPointer], plannerSize*sizeof(double));
//    memcpy(posMat2.data, (double *)[multiArr dataPointer] + plannerSize, plannerSize*sizeof(double));
//    memcpy(posMat3.data, (double *)[multiArr dataPointer] + plannerSize*2, plannerSize*sizeof(double));
//
//    vector<cv::Mat> posMats;
//    posMats.push_back(posMat1);
//    posMats.push_back(posMat2);
//    posMats.push_back(posMat3);
//
//    cv::Mat retMat;
//    cv::merge(posMats, retMat);
//    return retMat;
//}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection  {
    
    [connection setVideoOrientation:AVCaptureVideoOrientationPortrait];
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    int width = (int)CVPixelBufferGetWidth(imageBuffer);
    int height = (int)CVPixelBufferGetHeight(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    
    ////    ---MTCNN---     ////
    
    if(planerData==NULL) {
        planerData = (uint8_t *)malloc(width*height*3);
    }
    int cnt = 0;
    int planeSize = width*height;
    for(int i=0; i<width*height; i++) {
        planerData[planeSize*2 + cnt] = baseAddress[i*4];
        planerData[planeSize + cnt] = baseAddress[i*4+1];
        planerData[cnt] = baseAddress[i*4+2];
        cnt++;
    }
    
    cv::Mat chn[] = {
        cv::Mat(height, width, CV_8UC1, planerData),  // starting at 1st blue pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize),    // 1st green pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize*2)   // 1st red pixel
    };
    
    cv::Mat frame;
    merge(chn, 3, frame);
    
    cv::Mat outMat = [self renderTexture:frame :texture_color: output_image];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [[self imageView] setImage:[self UIImageFromCVMat:outMat]];
    });
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    
}

- (void)captureOutput2:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection  {

    [connection setVideoOrientation:AVCaptureVideoOrientationPortrait];
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);

    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    int width = (int)CVPixelBufferGetWidth(imageBuffer);
    int height = (int)CVPixelBufferGetHeight(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    
////    ---MTCNN---     ////
    
    if(planerData==NULL) {
        planerData = (uint8_t *)malloc(width*height*3);
    }
    int cnt = 0;
    int planeSize = width*height;
    for(int i=0; i<width*height; i++) {
        planerData[planeSize*2 + cnt] = baseAddress[i*4];
        planerData[planeSize + cnt] = baseAddress[i*4+1];
        planerData[cnt] = baseAddress[i*4+2];
        cnt++;
    }

    cv::Mat chn[] = {
        cv::Mat(height, width, CV_8UC1, planerData),  // starting at 1st blue pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize),    // 1st green pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize*2)   // 1st red pixel
    };

    cv::Mat frame;
    merge(chn, 3, frame);

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
    std::vector<Bbox> finalBbox;
    double start_time = CACurrentMediaTime();
    mtcnn.detect(ncnn_img, finalBbox);
    double finish_time = CACurrentMediaTime();
    double total_time = (double)(finish_time - start_time);
    std::cout << "cost " << total_time * 1000 << "ms" << std::endl;
    
    int num_box = (int)finalBbox.size();
    vector<uint32_t> realPos;
    if(num_box>0) {
        cv::Rect rect = cv::Rect(finalBbox[0].x1, finalBbox[0].y1, finalBbox[0].x2 - finalBbox[0].x1 + 1, finalBbox[0].y2 - finalBbox[0].y1 + 1);
        
        if(isInside(rect, cv::Rect(0,0,frame.cols,frame.rows))) {
            
            MLMultiArray *multiArr = [self facePRNetCoreML:frame(rect).clone()];
            
            int plannerSize = [[multiArr strides][0] intValue];
            
            for (int i=0; i<KPT_COUNT; i++) {
                
                int ind_y = face_data.uv_kpt_indices[i+KPT_COUNT];
                int ind_x = face_data.uv_kpt_indices[i];
                double u_data = *((double *)[multiArr dataPointer] + ind_y*256 + ind_x);
                double v_data = *((double *)[multiArr dataPointer] + plannerSize + ind_y*256 + ind_x);
                
                realPos.push_back(uint32_t(u_data*1.1*rect.width + rect.x));
                realPos.push_back(uint32_t(v_data*1.1*rect.width + rect.y));
            }
        }
    }

    cv::Mat show = drawKeyPoint(frame, realPos);
    
//    cv::Mat show = drawDetection(frame, finalBbox);
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [[self imageView] setImage:[self UIImageFromCVMat:show]];
        
    });
    
////    ---end MTCNN---     ////
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    
}

-(cv::Mat) renderTexture:(cv::Mat) targetMat :(float *) textureColor :(float *) outputImage {
    
    // textureColor ==> (rgb, rgb, rgb, ...)
    int w = targetMat.cols;
    int h = targetMat.rows;
    
    int faceCount = [self process:targetMat :posTemp];
    
    if(faceCount>0) {
        
        int count = 0;
        int nver = (int)face_data.face_indices.size();
        int ntri = (int)face_data.triangles.size()/3;
        
        // verticesTemp ==> (xyz, xyz, xyz, ...)
        for (int i=0; i<nver; i++) {
            int ind = face_data.face_indices[i];
            verticesTemp[count++] = float(*(posTemp+ ind));
            verticesTemp[count++] = float(*(posTemp+256*256   + ind));
            verticesTemp[count++] = float(*(posTemp+256*256*2 + ind));
        }
        
        int *triangles = (int *)face_data.triangles.data();

        for(int i=0; i<w*h; i++) {
            depth_buffer[i] = -999999.0;
        }

        
        float *vis_colors = (float *)malloc(nver*sizeof(float));
        for(int i=0; i<nver; i++) {
            vis_colors[i] = 1;
        }
        memset(face_mask, 0, 640*480*sizeof(float));
        
        // new_image ==> (rgb, rgb, rgb, ...)
        _render_colors_core(new_image, face_mask, verticesTemp, triangles, textureColor, vis_colors, depth_buffer, nver, ntri, h, w, 3);
        free(vis_colors);
        
        vector<cv::Mat> xc;
        split(targetMat, xc);
        
        cv::Mat mmm = cv::Mat(targetMat.rows, targetMat.cols, CV_32F, face_mask)*255;
        mmm.convertTo(mmm, CV_8U);
        cv::Rect rect = cv::boundingRect(mmm);
        
        count = 0;
        for(int i=0; i<w*h; i++) {
            outputImage[count++] = *(xc[2].data+i)*(1-face_mask[i]) + new_image[i*3+2]*face_mask[i]*255;
            outputImage[count++] = *(xc[1].data+i)*(1-face_mask[i]) + new_image[i*3+1]*face_mask[i]*255;
            outputImage[count++] = *(xc[0].data+i)*(1-face_mask[i]) + new_image[i*3]*face_mask[i]*255;
        }
        
        cv::Mat ooo = cv::Mat(targetMat.rows, targetMat.cols, CV_32FC3, outputImage);
        ooo.convertTo(ooo, CV_8U);
        
        cv::Mat outMat;
        cv::Mat maskMat = cv::Mat(targetMat.rows, targetMat.cols, CV_32F, face_mask)*255;
        maskMat.convertTo(maskMat, CV_8U);
        cv::seamlessClone(ooo, targetMat, maskMat, cv::Point(int(rect.x+rect.width/2), int(rect.y+rect.height/2)), outMat, cv::NORMAL_CLONE);
        
        return outMat;
    }
    return targetMat;
}

-(int) process: (cv::Mat) imgMat :(double *) pos{
    
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(imgMat.data, ncnn::Mat::PIXEL_BGR2RGB, imgMat.cols, imgMat.rows);
    std::vector<Bbox> finalBbox;

    mtcnn.detect(ncnn_img, finalBbox);
    
    int num_box = (int)finalBbox.size();
    vector<uint32_t> realPos;
    
    float maxScore=0;
    int maxProbIndex=0;
    for(int i=0; i<finalBbox.size(); i++) {
        if(finalBbox[i].score>maxScore) {
            maxScore = finalBbox[i].score;
            maxProbIndex = i;
        }
    }
    if(num_box>0) {
        int left = finalBbox[maxProbIndex].x1;
        int right = finalBbox[maxProbIndex].x2;
        int top = finalBbox[maxProbIndex].y1;
        int bottom = finalBbox[maxProbIndex].y2;
        
        float old_size = (right-left+bottom-top)/2.0;
        float centerX = right - (right-left)/2.0;
        float centerY = bottom - (bottom-top)/2 + old_size*0.14;
        int size = int(old_size*1.38);
        
        int x1 = centerX-size/2;
        int y1 = centerY-size/2;
        int x2 = centerX+size/2;
        int y2 = centerY+size/2;
        int width = x2 - x1;
        int height = y2 - y1;
        
        double scale = 256.0/width;
        double transX = -x1*scale;
        double transY = -y1*scale;
        
        if(x2>imgMat.cols) {
            cv::copyMakeBorder(imgMat, imgMat, 0, 0, 0, x2-imgMat.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        if(x1<0) {
            cv::copyMakeBorder(imgMat, imgMat, 0, 0, -x1, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
            x1 = 0;
        }
        if(y2>imgMat.rows) {
            cv::copyMakeBorder(imgMat, imgMat, 0, y2-imgMat.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        if(y1<0) {
            cv::copyMakeBorder(imgMat, imgMat, -y1, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
            y1 = 0;
        }
        cv::Mat cropped_image;
        cv::resize(imgMat(cv::Rect(x1, y1, width, height)), cropped_image, cv::Size(256,256));
        
        vector<cv::Mat> xc;
        split(cropped_image, xc);

        int count = 0;
        for(int i=0; i<256*256; i++) {
            inputData[count++] = *(xc[2].data+i)/256.0;
        }
        for(int i=0; i<256*256; i++) {
            inputData[count++] = *(xc[1].data+i)/256.0;
        }
        for(int i=0; i<256*256; i++) {
            inputData[count++] = *(xc[0].data+i)/256.0;
        }

        MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:inputData shape:[NSArray arrayWithObjects:[NSNumber numberWithInt:3], [NSNumber numberWithInt:256], [NSNumber numberWithInt:256], nil] dataType:MLMultiArrayDataTypeDouble strides:[NSArray arrayWithObjects:[NSNumber numberWithInt:256*256], [NSNumber numberWithInt:256], [NSNumber numberWithInt:1], nil] deallocator:nil error:nil];

        prnetOutput *output = [irModel predictionFromPlaceholder__0:arr error:nil];
        MLMultiArray *multiArr = [output resfcn256__Conv2d_transpose_16__Sigmoid__0];

        int plannerSize = [[multiArr strides][0] intValue];
        double *dataPointer = (double *)[multiArr dataPointer];
        for(int i=0; i<plannerSize*3; i++) {
            dataPointer[i] *= 1.1*256;
        }
        
        cv::Mat posMat1(1,256*256,CV_64F, dataPointer);
        cv::Mat posMat2(1,256*256,CV_64F, dataPointer + plannerSize);
        cv::Mat posMat3(1,256*256,CV_64F, dataPointer + plannerSize*2);

        double tformData[9] = {scale,0.0,transX, 0.0,scale,transY, 0.0,0.0,1.0};
        cv::Mat tform(3,3,CV_64F, tformData);
        cv::Mat z = posMat3/scale;
        posMat3.setTo(cv::Scalar(1));

        cv::Mat posMats;
        posMats.push_back(posMat1);
        posMats.push_back(posMat2);
        posMats.push_back(posMat3);

        cv::Mat vertices;
        vertices = tform.inv()*posMats;
        z.row(0).copyTo(vertices.row(2));

        memcpy(pos, vertices.data, 256*256*3*sizeof(double));
        
        return 1;
    }
    return 0;
}

-(void)startCapture:(UIImageView *)capImageView {
    
    NSArray *cameraArray = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in cameraArray) {
        if ([device position] == AVCaptureDevicePositionBack) {
            inputDevice = [AVCaptureDeviceInput deviceInputWithDevice:device error:nil];
        }
    }

    session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPreset640x480;
    [session addInput:inputDevice];     //输入设备与session连接
    
    /*  设置输出yuv格式   */
    output = [[AVCaptureVideoDataOutput alloc] init];
    NSNumber *value = [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA];
    NSDictionary *dictionary = [NSDictionary dictionaryWithObject:value forKey:(NSString *)kCVPixelBufferPixelFormatTypeKey];
    [output setVideoSettings:dictionary];
    [output setAlwaysDiscardsLateVideoFrames:YES];
    
    /*  设置输出回调队列    */
    dispatch_queue_t queue = dispatch_queue_create("com.linku.queue", NULL);
    [output setSampleBufferDelegate:self queue:queue];
    //    dispatch_release(queue);
    [session addOutput:output];     //输出与session连接
    
    /////////////mtcnn-ncnn
    char *path = (char *)[[[NSBundle mainBundle] resourcePath] UTF8String];
    mtcnn.init(path);
    mtcnn.SetMinFace(40);
    planerData = NULL;

    irModel = [[prnet alloc] init];
    inputData = (double *)malloc(sizeof(double)*256*256*3);
    
    LoadFaceData([[[NSBundle mainBundle] resourcePath] UTF8String], &face_data);
    
    
    cv::Mat imgMat = cv::imread([[[NSBundle mainBundle] pathForResource:@"ref" ofType:@"jpg"] UTF8String]);
    posTemp = (double *)malloc(256*256*3*sizeof(double));
    verticesTemp = (float *)malloc(256*256*3*sizeof(float));
    new_image = (float *)malloc(640*480*3*sizeof(float));
    face_mask = (float *)malloc(640*480*sizeof(float));
    output_image = (float *)malloc(640*480*3*sizeof(float));
    
    memset(new_image, 0, 640*480*3*sizeof(float));
    memset(face_mask, 0, 640*480*sizeof(float));
    memset(output_image, 0, 640*480*3*sizeof(float));
    
    depth_buffer = (float *)malloc(640*480*sizeof(float));
    double *pos = (double *)malloc(256*256*3*sizeof(double));
    int faceCount = [self process:imgMat :pos]; // pos ==> (rrrrr..., ggggg..., bbbbb...)
    if(faceCount>0) {
        
        cv::Mat ref_pos1 = cv::Mat(256,256, CV_64F, pos);
        cv::Mat ref_pos2 = cv::Mat(256,256, CV_64F, pos + 256*256);
        
        cv::Mat posMat;
        vector<cv::Mat> posMats;
        posMats.push_back(ref_pos1);
        posMats.push_back(ref_pos2);
        cv::merge(posMats, posMat);
        
        posMat.convertTo(posMat, CV_32FC2);
        imgMat.convertTo(imgMat, CV_32FC3, 1/256.0);
        
        cv::Mat remapMat;
        cv::remap(imgMat, remapMat, posMat, cv::Mat(), cv::INTER_NEAREST);
        
        texture_color = (float *)malloc(face_data.face_indices.size()*sizeof(float)*3);
        int count = 0;
        vector<cv::Mat> xc2;
        split(remapMat, xc2);
        
        for (int i=0; i<face_data.face_indices.size(); i++) {
            int ind = face_data.face_indices[i];
            texture_color[count++] = *((float *)xc2[0].data+ ind);
            texture_color[count++] = *((float *)xc2[1].data+ ind);
            texture_color[count++] = *((float *)xc2[2].data+ ind);
        }
    }
    
    [session startRunning];
    isCapture = true;
    
}

@end
