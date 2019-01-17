//
//  ViewController.m
//  FaceConverter
//
//  Created by LuDong on 2019/1/9.
//  Copyright © 2019年 LuDong. All rights reserved.
//

#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    [[self imageView] setFrame:[[UIScreen mainScreen] bounds]];
    [[self imageView] setContentMode:UIViewContentModeScaleAspectFit];
    [self startCapture:[self imageView]];
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
