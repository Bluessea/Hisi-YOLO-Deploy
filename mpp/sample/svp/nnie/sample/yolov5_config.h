#ifndef FD_CONFIG
#define FD_CONFIG

#include <stdio.h>
//#include"list.h"
#include<math.h>
#include "sample_comm_nnie.h"

#define QUANT_BASE 4096.0f

#define yolo_layer_num 3 // yolo layer 层数

#ifndef YOLO_MIN
#define YOLO_MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef YOLO_MAX
#define YOLO_MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define IMAGE_W 448.0f // 输入图片大小

#define IMAGE_H 448.0f

typedef struct Anchor_W_H
{
	float anchor_w;
	
	float anchor_h;
}Anchor_W_H;

typedef struct Yolov5_Result
{
	float left_up_x;
	
	float left_up_y;

	float right_down_x;

	float right_down_y;

	int class_index;

	float score;

	struct yolo_result* next;
}Yolov5_Result;

Anchor_W_H  anchor_grids[3][3] = {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}, // small yolo layer 层 anchor

									{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}, // middle yolo layer 层 anchor

									{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}}; // large yolo layer 层 anchor
	
float strides[3] = {8.0f, 16.0f, 32.0f}; // 每个 yolo 层，grid 大小，与上面顺序对应

int map_size[3] = {56, 28, 14}; // 每个 yolo 层，feature map size 大小，与上面顺序对应


#endif // FD_CONFIG
