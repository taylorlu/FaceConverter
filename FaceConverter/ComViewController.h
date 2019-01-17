//
//  ComViewController.h
//  VideoFace
//
//  Created by LuDong on 2018/7/2.
//  Copyright © 2018年 LuDong. All rights reserved.
//


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#import <AVFoundation/AVFoundation.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <Endian.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>
#include "mtcnn.h"
#include "net.h"
#import "prnet.h"
#include "face-data.h"
#include "mesh_core.h"
#import <UIKit/UIKit.h>
#define KPT_COUNT 68

@interface ComViewController : UIViewController<AVCaptureVideoDataOutputSampleBufferDelegate> {
    
    bool isCapture;
    uint8_t *originData;
    
    AVCaptureVideoDataOutput *output;
    AVCaptureSession     *session;
    AVCaptureDeviceInput *inputDevice;
    AVCaptureVideoPreviewLayer   *previewLayer;
    
    CIDetector * faceDetector;
    CIDetector * textDetector;
    VNDetectFaceLandmarksRequest *faceLandmarks;
    
    VNDetectFaceRectanglesRequest *faceRectangles;
    VNSequenceRequestHandler *faceSequenceRequest;
    
    VNDetectTextRectanglesRequest *textRectangles;
    VNSequenceRequestHandler *textSequenceRequest;
    
    MTCNN mtcnn;
    prnet *irModel;
    float *faceVector;
    uint8_t *planerData;
    ncnn::Net faceNet;
    cv::Mat resultMat;
    float *baseMatrix;
    double *inputData;
    double *posTemp;
    float *verticesTemp;
    float *new_image;
    float *face_mask;
    float *output_image;
    float *depth_buffer;
    
    float *texture_color;
    
    FaceData face_data;
}

@property(weak, nonatomic) IBOutlet UIImageView *imageView;

-(void)startCapture:(UIImageView *)capImageView;

@end
