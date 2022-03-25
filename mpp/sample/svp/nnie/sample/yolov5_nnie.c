#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "yolov5_config.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "sample_nnie_main.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"


/*cnn para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stCnnModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stCnnNnieParam = {0};
static SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S s_stCnnSoftwareParam = {0};
/*segment para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stSegnetModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stSegnetNnieParam = {0};
/*fasterrcnn para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stFasterRcnnModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stFasterRcnnNnieParam = {0};
static SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S s_stFasterRcnnSoftwareParam = {0};
static SAMPLE_SVP_NNIE_NET_TYPE_E s_enNetType;
/*rfcn para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stRfcnModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stRfcnNnieParam = {0};
static SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S s_stRfcnSoftwareParam = {0};
static SAMPLE_IVE_SWITCH_S s_stRfcnSwitch = {HI_FALSE,HI_FALSE};
static HI_BOOL s_bNnieStopSignal = HI_FALSE;
static pthread_t s_hNnieThread = 0;
static SAMPLE_VI_CONFIG_S s_stViConfig = {0};

/*ssd para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stSsdModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stSsdNnieParam = {0};
static SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S s_stSsdSoftwareParam = {0};
/*yolov1 para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov1Model = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov1NnieParam = {0};
static SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S s_stYolov1SoftwareParam = {0};
/*yolov2 para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov2Model = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov2NnieParam = {0};
static SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S s_stYolov2SoftwareParam = {0};
/*yolov3 para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov3Model = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov3NnieParam = {0};
static SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S s_stYolov3SoftwareParam = {0};

/*yolov3 real time para*/
static SAMPLE_IVE_SWITCH_S s_stYolov3Switch = {HI_FALSE,HI_FALSE};

/**
 * @brief yolov5 para*
 *  2022/3/11
 */
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov5Model = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov5NnieParam = {0};
static SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S s_stYolov5SoftwareParam = {0};

/*yolov5 real time para*/
static SAMPLE_IVE_SWITCH_S s_stYolov5Switch = {HI_FALSE,HI_FALSE};

/*lstm para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stLstmModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stLstmNnieParam = {0};
/*pvanet para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stPvanetModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stPvanetNnieParam = {0};
static SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S s_stPvanetSoftwareParam = {0};

/**
 * @brief sigmoid fun
 * 
 * @param x 
 * @return float 
 */
float sigmoid(float x){
	return (1.0f / ((float)exp((double)(-x)) + 1.0f));
}


#ifdef SAMPLE_SVP_NNIE_PERF_STAT
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()  memset(&s_stOpForwardPerfTmp,0,sizeof(s_stOpForwardPerfTmp));
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64SrcFlushTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64PreDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64AferDstFulshTime)
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_GET_DIFF_TIME(s_stOpForwardPerfTmp.u64OPTime)

/*YoloV1*/
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_SRC_FLUSH_TIME() s_stYolov1Perf.stForwardPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_PRE_DST_FLUSH_TIME() s_stYolov1Perf.stForwardPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_AFTER_DST_FLUSH_TIME() s_stYolov1Perf.stForwardPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_OP_TIME() s_stYolov1Perf.stForwardPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov1Perf.stGRPerf.u64SrcFlushTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov1Perf.stGRPerf.u64PreDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov1Perf.stGRPerf.u64AferDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov1Perf.stGRPerf.u64OPTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_PRINT() printf("Yolov1 Forward time: %llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stYolov1Perf.stForwardPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stYolov1Perf.stGRPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    (s_stYolov1Perf.stForwardPerf.u64SrcFlushTime + s_stYolov1Perf.stForwardPerf.u64PreDstFulshTime + s_stYolov1Perf.stForwardPerf.u64AferDstFulshTime\
    + s_stYolov1Perf.stGRPerf.u64SrcFlushTime + s_stYolov1Perf.stGRPerf.u64PreDstFulshTime + s_stYolov1Perf.stGRPerf.u64AferDstFulshTime)/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES );
/*Yolov2*/
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_SRC_FLUSH_TIME() s_stYolov2Perf.stForwardPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_PRE_DST_FLUSH_TIME() s_stYolov2Perf.stForwardPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_AFTER_DST_FLUSH_TIME() s_stYolov2Perf.stForwardPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_OP_TIME() s_stYolov2Perf.stForwardPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov2Perf.stGRPerf.u64SrcFlushTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov2Perf.stGRPerf.u64PreDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov2Perf.stGRPerf.u64AferDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stYolov2Perf.stGRPerf.u64OPTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_PRINT() printf("Yolov2 Forward time: %llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stYolov2Perf.stForwardPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stYolov2Perf.stGRPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    (s_stYolov2Perf.stForwardPerf.u64SrcFlushTime + s_stYolov2Perf.stForwardPerf.u64PreDstFulshTime + s_stYolov2Perf.stForwardPerf.u64AferDstFulshTime\
    + s_stYolov2Perf.stGRPerf.u64SrcFlushTime + s_stYolov2Perf.stGRPerf.u64PreDstFulshTime + s_stYolov2Perf.stGRPerf.u64AferDstFulshTime)/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES );

/*SSD*/
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_SRC_FLUSH_TIME() s_stSsdPerf.stForwardPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_PRE_DST_FLUSH_TIME() s_stSsdPerf.stForwardPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_AFTER_DST_FLUSH_TIME() s_stSsdPerf.stForwardPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_OP_TIME() s_stSsdPerf.stForwardPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stSsdPerf.stGRPerf.u64SrcFlushTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stSsdPerf.stGRPerf.u64PreDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stSsdPerf.stGRPerf.u64AferDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stSsdPerf.stGRPerf.u64OPTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_PRINT() printf("SSD Forward time: %llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stSsdPerf.stForwardPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stSsdPerf.stGRPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    (s_stSsdPerf.stForwardPerf.u64SrcFlushTime + s_stSsdPerf.stForwardPerf.u64PreDstFulshTime + s_stSsdPerf.stForwardPerf.u64AferDstFulshTime\
    + s_stSsdPerf.stGRPerf.u64SrcFlushTime + s_stSsdPerf.stGRPerf.u64PreDstFulshTime + s_stSsdPerf.stGRPerf.u64AferDstFulshTime)/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES );

/*Pvanet*/
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_SRC_FLUSH_TIME() s_stPvanetPerf.stForwardPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_PRE_DST_FLUSH_TIME() s_stPvanetPerf.stForwardPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_AFTER_DST_FLUSH_TIME() s_stPvanetPerf.stForwardPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_OP_TIME() s_stPvanetPerf.stForwardPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_SRC_FLUSH_TIME() s_stPvanetPerf.stRpnPerf.u64SrcFlushTime += g_stOpRpnPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_PRE_DST_FLUSH_TIME() s_stPvanetPerf.stRpnPerf.u64PreDstFulshTime += g_stOpRpnPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_AFTER_DST_FLUSH_TIME() s_stPvanetPerf.stRpnPerf.u64AferDstFulshTime += g_stOpRpnPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_OP_TIME() s_stPvanetPerf.stRpnPerf.u64OPTime += g_stOpRpnPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_SRC_FLUSH_TIME() s_stPvanetPerf.stRoiPoolingPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_PRE_DST_FLUSH_TIME() s_stPvanetPerf.stRoiPoolingPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_AFTER_DST_FLUSH_TIME() s_stPvanetPerf.stRoiPoolingPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_OP_TIME() s_stPvanetPerf.stRoiPoolingPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stPvanetPerf.stGRPerf.u64SrcFlushTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stPvanetPerf.stGRPerf.u64PreDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stPvanetPerf.stGRPerf.u64AferDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stPvanetPerf.stGRPerf.u64OPTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_PRINT() printf("Pvanet Forward time: %llu us,Rpn time:%llu us,RoiPooling time:%llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stPvanetPerf.stForwardPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stPvanetPerf.stRpnPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    s_stPvanetPerf.stRoiPoolingPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES, s_stPvanetPerf.stGRPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    (s_stPvanetPerf.stForwardPerf.u64SrcFlushTime + s_stPvanetPerf.stForwardPerf.u64PreDstFulshTime + s_stPvanetPerf.stForwardPerf.u64AferDstFulshTime\
    + s_stPvanetPerf.stRpnPerf.u64SrcFlushTime + s_stPvanetPerf.stRpnPerf.u64PreDstFulshTime + s_stPvanetPerf.stRpnPerf.u64AferDstFulshTime\
    + s_stPvanetPerf.stRoiPoolingPerf.u64SrcFlushTime + s_stPvanetPerf.stRoiPoolingPerf.u64PreDstFulshTime + s_stPvanetPerf.stRoiPoolingPerf.u64AferDstFulshTime\
    + s_stPvanetPerf.stGRPerf.u64SrcFlushTime + s_stPvanetPerf.stGRPerf.u64PreDstFulshTime + s_stPvanetPerf.stGRPerf.u64AferDstFulshTime)/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES );

/*RFCN*/
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_SRC_FLUSH_TIME() s_stRfcnPerf.stForwardPerf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_PRE_DST_FLUSH_TIME() s_stRfcnPerf.stForwardPerf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_AFTER_DST_FLUSH_TIME() s_stRfcnPerf.stForwardPerf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_OP_TIME() s_stRfcnPerf.stForwardPerf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_SRC_FLUSH_TIME() s_stRfcnPerf.stRpnPerf.u64SrcFlushTime += g_stOpRpnPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_PRE_DST_FLUSH_TIME() s_stRfcnPerf.stRpnPerf.u64PreDstFulshTime += g_stOpRpnPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_AFTER_DST_FLUSH_TIME() s_stRfcnPerf.stRpnPerf.u64AferDstFulshTime += g_stOpRpnPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_OP_TIME() s_stRfcnPerf.stRpnPerf.u64OPTime += g_stOpRpnPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_SRC_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling1Perf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_PRE_DST_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling1Perf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_AFTER_DST_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling1Perf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_OP_TIME() s_stRfcnPerf.stPsRoiPooling1Perf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_SRC_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling2Perf.u64SrcFlushTime += s_stOpForwardPerfTmp.u64SrcFlushTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_PRE_DST_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling2Perf.u64PreDstFulshTime += s_stOpForwardPerfTmp.u64PreDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_AFTER_DST_FLUSH_TIME() s_stRfcnPerf.stPsRoiPooling2Perf.u64AferDstFulshTime += s_stOpForwardPerfTmp.u64AferDstFulshTime;
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_OP_TIME() s_stRfcnPerf.stPsRoiPooling2Perf.u64OPTime += s_stOpForwardPerfTmp.u64OPTime;

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_SRC_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stRfcnPerf.stGRPerf.u64SrcFlushTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_PRE_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stRfcnPerf.stGRPerf.u64PreDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_AFTER_DST_FLUSH_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stRfcnPerf.stGRPerf.u64AferDstFulshTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_OP_TIME() SAMPLE_SVP_NNIE_PERF_STAT_ADD_DIFF_TIME(s_stRfcnPerf.stGRPerf.u64OPTime);
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PRINT_VITOVO() printf("Rfcn_ViToVo Forward time: %llu us,Rpn time:%llu us,PsRoiPooling1 time:%llu us,PsRoiPooling2 time:%llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stRfcnPerf.stForwardPerf.u64OPTime,s_stRfcnPerf.stRpnPerf.u64OPTime,\
    s_stRfcnPerf.stPsRoiPooling1Perf.u64OPTime, s_stRfcnPerf.stPsRoiPooling2Perf.u64OPTime,s_stRfcnPerf.stGRPerf.u64OPTime,\
    (s_stRfcnPerf.stForwardPerf.u64SrcFlushTime + s_stRfcnPerf.stForwardPerf.u64PreDstFulshTime + s_stRfcnPerf.stForwardPerf.u64AferDstFulshTime\
    + s_stRfcnPerf.stRpnPerf.u64SrcFlushTime + s_stRfcnPerf.stRpnPerf.u64PreDstFulshTime + s_stRfcnPerf.stRpnPerf.u64AferDstFulshTime\
    + s_stRfcnPerf.stPsRoiPooling1Perf.u64SrcFlushTime + s_stRfcnPerf.stPsRoiPooling1Perf.u64PreDstFulshTime + s_stRfcnPerf.stPsRoiPooling1Perf.u64AferDstFulshTime\
    + s_stRfcnPerf.stPsRoiPooling2Perf.u64SrcFlushTime + s_stRfcnPerf.stPsRoiPooling2Perf.u64PreDstFulshTime + s_stRfcnPerf.stPsRoiPooling2Perf.u64AferDstFulshTime\
    + s_stRfcnPerf.stGRPerf.u64SrcFlushTime + s_stRfcnPerf.stGRPerf.u64PreDstFulshTime + s_stRfcnPerf.stGRPerf.u64AferDstFulshTime));

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PRINT_FILE() printf("Rfcn_File Forward time: %llu us,Rpn time:%llu us,PsRoiPooling1 time:%llu us,PsRoiPooling2 time:%llu us,GR time:%llu us,Flush time: %llu us\n",\
    s_stRfcnPerf.stForwardPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stRfcnPerf.stRpnPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    s_stRfcnPerf.stPsRoiPooling1Perf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,s_stRfcnPerf.stPsRoiPooling2Perf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    s_stRfcnPerf.stGRPerf.u64OPTime/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES,\
    (s_stRfcnPerf.stForwardPerf.u64SrcFlushTime + s_stRfcnPerf.stForwardPerf.u64PreDstFulshTime + s_stRfcnPerf.stForwardPerf.u64AferDstFulshTime\
    + s_stRfcnPerf.stRpnPerf.u64SrcFlushTime + s_stRfcnPerf.stRpnPerf.u64PreDstFulshTime + s_stRfcnPerf.stRpnPerf.u64AferDstFulshTime\
    + s_stRfcnPerf.stPsRoiPooling1Perf.u64SrcFlushTime + s_stRfcnPerf.stPsRoiPooling1Perf.u64PreDstFulshTime + s_stRfcnPerf.stPsRoiPooling1Perf.u64AferDstFulshTime\
    + s_stRfcnPerf.stPsRoiPooling2Perf.u64SrcFlushTime + s_stRfcnPerf.stPsRoiPooling2Perf.u64PreDstFulshTime + s_stRfcnPerf.stPsRoiPooling2Perf.u64AferDstFulshTime\
    + s_stRfcnPerf.stGRPerf.u64SrcFlushTime + s_stRfcnPerf.stGRPerf.u64PreDstFulshTime + s_stRfcnPerf.stGRPerf.u64AferDstFulshTime)/SAMPLE_SVP_NNIE_PERF_STAT_LOOP_TIMES);



static SAMPLE_SVP_NNIE_YOLO_PERF_STAT_S s_stYolov1Perf = {0};
static SAMPLE_SVP_NNIE_YOLO_PERF_STAT_S s_stYolov2Perf = {0};
static SAMPLE_SVP_NNIE_SSD_PERF_STAT_S  s_stSsdPerf = {0};
static SAMPLE_SVP_NNIE_PVANET_PERF_STAT_S s_stPvanetPerf = {0};
static SAMPLE_SVP_NNIE_RFCN_PERF_STAT_S s_stRfcnPerf = {0};

static SAMPLE_SVP_NNIE_OP_PERF_STAT_S   s_stOpForwardPerfTmp = {0};
extern SAMPLE_SVP_NNIE_OP_PERF_STAT_S   g_stOpRpnPerfTmp;
#else

#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_FORWARD_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_GR_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV1_PRINT()

/*Yolov2*/
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_FORWARD_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_GR_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_YOLOV2_PRINT()
/*SSD*/
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_FORWARD_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_GR_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_SSD_PRINT()

/*Pvanet*/
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_FORWARD_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_RPN_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_ROIPOOLING_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_GR_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_PVANET_PRINT()

/*RFCN*/
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_FORWARD_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_RPN_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING1_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PSROIPOOLING2_OP_TIME()

#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_SRC_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_PRE_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_AFTER_DST_FLUSH_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_GR_OP_TIME()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PRINT_VITOVO()
#define SAMPLE_SVP_NNIE_PERF_STAT_RFCN_PRINT_FILE()

#endif

/******************************************************************************
* function : NNIE Forward
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    SAMPLE_SVP_NIE_PERF_STAT_DEF_VAR()

    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

   
    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME()

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx,i);
        }
    }

    // SAMPLE_SVP_TRACE_INFO("The s_stYolov5ModelF Height is %d\n", pstNnieParam->astSegData->astDst->unShape.stWhc.u32Height);
    // SAMPLE_SVP_TRACE_INFO("The s_stYolov5ModelFWidth is %d\n", pstNnieParam->astSegData->astDst->unShape.stWhc.u32Width);

    
    /*NNIE_Forward*/
    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if(bInstant)
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME()
    u32TotalStepNum = 0;


    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME()

    return s32Ret;
}

/******************************************************************************
* function : NNIE ForwardWithBbox
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_ForwardWithBbox(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,SVP_SRC_BLOB_S astBbox[],
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    HI_U32 i, j;
    SAMPLE_SVP_NIE_PERF_STAT_DEF_VAR()

    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_CLREAR()

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()

    for(i = 0; i < pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_PRE_DST_FLUSH_TIME()

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx,i);
        }
    }
    /*NNIE_ForwardWithBbox*/

    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&hSvpNnieHandle,
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,astBbox,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_ForwardWithBbox failed!\n");

    if(bInstant)
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_OP_TIME()

    u32TotalStepNum = 0;
    SAMPLE_SVP_NNIE_PERF_STAT_BEGIN()

    for(i = 0; i < pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
        else
        {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }
    SAMPLE_SVP_NNIE_PERF_STAT_END()
    SAMPLE_SVP_NNIE_PERF_STAT_OP_FORWARD_AFTER_DST_FLUSH_TIME()

    return s32Ret;
}

/******************************************************************************
* function : Yolov3 software deinit
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(NULL== pstSoftWareParam,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if(0!=pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr && 0!=pstSoftWareParam->stGetResultTmpBuf.u64VirAddr)
    {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}


/******************************************************************************
* function : Fill Src Data
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx)
{
    FILE* fp = NULL;
    HI_U32 i =0, j = 0, n = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U8*pu8PicAddr = NULL;
    HI_U32*pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;

    /*open file*/
    if (NULL != pstNnieCfg->pszPic)
    {
        fp = fopen(pstNnieCfg->pszPic,"rb");
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error, open file failed!\n");
    }

    /*get data size*/
    if(SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    /*fill src data*/
    if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for(i = 0;i < *(pu32StepAddr+n); i++)
            {
                s32Ret = fread(pu8PicAddr,u32Dim*u32VarSize,1,fp);
                SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr+n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            u32TotalStepNum*u32Stride);
    }
    else
    {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        if(SVP_BLOB_TYPE_YVU420SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Chn*u32Height/2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else if(SVP_BLOB_TYPE_YVU422SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Height*2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0;i < u32Chn; i++)
                {
                    for(j = 0; j < u32Height; j++)
                    {
                        s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                        SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                        pu8PicAddr += u32Stride;
                    }
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num*u32Chn*u32Height*u32Stride);
    }

    fclose(fp);
    return HI_SUCCESS;
FAIL:

    fclose(fp);
    return HI_FAILURE;
}

/******************************************************************************
* function : Yolov3 software para init
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;

    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 3;
    pstSoftWareParam->u32ClassNum = 2;    // ClassName
    pstSoftWareParam->au32GridNumHeight[0] = 14;
    pstSoftWareParam->au32GridNumHeight[1] = 28;
    pstSoftWareParam->au32GridNumHeight[2] = 56;
    pstSoftWareParam->au32GridNumWidth[0] = 14;
    pstSoftWareParam->au32GridNumWidth[1] = 28;
    pstSoftWareParam->au32GridNumWidth[2] = 56;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f*SAMPLE_SVP_NNIE_QUANT_BASE);    //0.3
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.5f*SAMPLE_SVP_NNIE_QUANT_BASE);   //0.5
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0][0] = 116;
    pstSoftWareParam->af32Bias[0][1] = 90;
    pstSoftWareParam->af32Bias[0][2] = 156;
    pstSoftWareParam->af32Bias[0][3] = 198;
    pstSoftWareParam->af32Bias[0][4] = 373;
    pstSoftWareParam->af32Bias[0][5] = 326;
    pstSoftWareParam->af32Bias[1][0] = 30;
    pstSoftWareParam->af32Bias[1][1] = 61;
    pstSoftWareParam->af32Bias[1][2] = 62;
    pstSoftWareParam->af32Bias[1][3] = 45;
    pstSoftWareParam->af32Bias[1][4] = 59;
    pstSoftWareParam->af32Bias[1][5] = 119;
    pstSoftWareParam->af32Bias[2][0] = 10;
    pstSoftWareParam->af32Bias[2][1] = 13;
    pstSoftWareParam->af32Bias[2][2] = 16;
    pstSoftWareParam->af32Bias[2][3] = 30;
    pstSoftWareParam->af32Bias[2][4] = 33;
    pstSoftWareParam->af32Bias[2][5] = 23;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSoftWareParam->u32ClassNum+1;

    SAMPLE_SVP_CHECK_EXPR_RET(SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM != pstNnieParam->pstModel->astSeg[0].u16DstNum,
        HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,pstNnieParam->pstModel->astSeg[0].u16DstNum(%d) should be %d!\n",
        pstNnieParam->pstModel->astSeg[0].u16DstNum,SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM);
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf(pstNnieParam,pstSoftWareParam);
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*pstSoftWareParam->u32MaxRoiNum*sizeof(HI_U32)*SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*pstSoftWareParam->u32MaxRoiNum*sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*sizeof(HI_U32));
    u32TotalSize = u32TotalSize+u32DstRoiSize+u32DstScoreSize+u32ClassRoiNumSize+u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV3_INIT",NULL,(HI_U64*)&u64PhyAddr,
        (void**)&pu8VirAddr,u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    memset(pu8VirAddr,0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr,(void*)pu8VirAddr,u32TotalSize);

   /*set each tmp buffer addr*/
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr);

    /*set result blob*/
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr+u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*
        pstSoftWareParam->u32MaxRoiNum*sizeof(HI_U32)*SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum*
        pstSoftWareParam->u32MaxRoiNum*SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize+u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr+u32TmpBufTotalSize+u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*
        pstSoftWareParam->u32MaxRoiNum*sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum*pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize+
        u32DstRoiSize+u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr+u32TmpBufTotalSize+
        u32DstRoiSize+u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}


/******************************************************************************
* function : Yolov5 Deinit
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov5_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftWareParam,SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*hardware deinit*/
    if(pstNnieParam!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*software deinit*/
    
    if(pstSoftWareParam!=NULL)
    {
        s32Ret = SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit failed!\n");
    }
    
    /*model deinit*/
    if(pstNnieModel!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/******************************************************************************
* function : Yolov5 init
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov5_ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;

    /*init hardware para*/
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg,pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,INIT_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n",s32Ret);

    /*init software para*/
    // s32Ret = SAMPLE_SVP_NNIE_Yolov3_SoftwareInit(pstCfg,pstNnieParam,
    //     pstSoftWareParam);
    // SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,INIT_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //     "Error(%#x),SAMPLE_SVP_NNIE_Yolov3_SoftwareInit failed!\n",s32Ret);
    
    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Yolov5_Deinit(pstNnieParam,pstSoftWareParam,NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SAMPLE_SVP_NNIE_Yolov5_Deinit failed!\n",s32Ret);
    return HI_FAILURE;

}

/**
 * @brief Yolov5 getResult
 * 
 */
static unsigned int Yolov5_GetResult(SAMPLE_SVP_NNIE_PARAM_S*pstNnieParam,float *strides, Anchor_W_H (*anchor_grids)[3], int *map_size, Yolov5_Result **output_result, float confidence_threshold)
{
    HI_S32 output_num = 0;
    HI_S32 anchor_num = 0;
    HI_S32 feature_length = 0;
    float anchor_w = 0.0f;
    float anchor_h = 0.0f;
    int x = 0;
    int y = 0;
    HI_S32* output_addr = NULL;
    float confidence = 0.0f;
    float class_confidence = 0.0f;

	//float confidence_threshold = 0.4f;

	float pred_x = 0.0f;

	float pred_y = 0.0f;

	float pred_w = 0.0f;

	float pred_h = 0.0f;

	Yolov5_Result *current = NULL;
	
	Yolov5_Result *former = NULL;
	
	*output_result = NULL;

	unsigned int resltu_num = 0;
	// 取数据这里，我写篇博客说明一下，之前卡了我一会，觉得需要记录一下，帮助自己也帮助他人
    for (int yolo_layer_index = 0; yolo_layer_index < yolo_layer_num; yolo_layer_index ++) { // 3 yolo layer 
    
		feature_length = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Width; // 1600 / 400 / 100
		
		output_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Height; // 53
		
		anchor_num = pstNnieParam->astSegData[0].astDst[yolo_layer_index].unShape.stWhc.u32Chn; // 3
		
		output_addr = (HI_S32* )((HI_U8* )pstNnieParam->astSegData[0].astDst[yolo_layer_index].u64VirAddr); // yolo 输出的首地址
		
		for (int anchor_index = 0; anchor_index < anchor_num; anchor_index ++){ // 每个 grid 上有三个 anchor
				
				anchor_w = anchor_grids[yolo_layer_index][anchor_index].anchor_w;
				anchor_h = anchor_grids[yolo_layer_index][anchor_index].anchor_h;
				
				for (int coord_x_y = 0; coord_x_y < feature_length; coord_x_y ++){ // feature size 拉直后的长度，如 1600 400 100.
					y = coord_x_y / map_size[yolo_layer_index];
					x = coord_x_y % map_size[yolo_layer_index];
					
					confidence = *(output_addr + anchor_index * feature_length * output_num + 4 * feature_length + coord_x_y) / 4096.0f;  // confidence
					confidence = sigmoid(confidence);
					
					if (confidence > confidence_threshold){
						
						for (int output_index = 5; output_index < output_num; output_index ++){
							class_confidence = *(output_addr + anchor_index * feature_length * output_num + output_index * feature_length + coord_x_y) / 4096.0f;  // class confidence
							class_confidence = sigmoid(class_confidence) * confidence;
							// 注意，yolo v5 的类别置信度并不需要选出个最大值，它的 label 是多标签，所以并不是 softmax，我在博客里说明一下
							if (class_confidence > confidence_threshold){
								
                                /**
                                 * @brief Net output memory distribution
                                 * height : class nums for example VOC 20/COCO 80
                                 *              width = size for example 52*52/26*26/13*13 
                                 * anchor0  .  s32_10 s32_20 s32_30 ........
                                 *    height.  s32_10 s32_20 s32_30 ........
                                 *          .  s32_10 s32_20 s32_30 ........
                                 *             .      .      .
                                 *             .      .      .
                                 *             .      .      .
                                 * anchor1  .  s32_10 s32_20 s32_30 ........
                                 *    height.  s32_10 s32_20 s32_30 ........
                                 *          .  s32_10 s32_20 s32_30 ........
                                 */
								pred_x = *(output_addr + anchor_index * feature_length * output_num + 0 * feature_length + coord_x_y) / 4096.0f; // x
								pred_y = *(output_addr + anchor_index * feature_length * output_num + 1 * feature_length + coord_x_y) / 4096.0f; // y
								pred_w = *(output_addr + anchor_index * feature_length * output_num + 2 * feature_length + coord_x_y) / 4096.0f; // w
								pred_h = *(output_addr + anchor_index * feature_length * output_num + 3 * feature_length + coord_x_y) / 4096.0f; // h 

								pred_x = sigmoid(pred_x);
								pred_y = sigmoid(pred_y);
								pred_w = sigmoid(pred_w);
								pred_h = sigmoid(pred_h);
								// bbox 输出结果处理
								pred_x = (pred_x * 2.0f - 0.5f + (float)x) * strides[yolo_layer_index];
								pred_y = (pred_y * 2.0f - 0.5f + (float)y) * strides[yolo_layer_index];
								pred_w = (pred_w * 2.0f) * (pred_w * 2.0f) * anchor_w;
								pred_h = (pred_h * 2.0f) * (pred_h * 2.0f) * anchor_h;
								
								current = (Yolov5_Result *) malloc(sizeof(Yolov5_Result));
								// 坐标转换 (x y w h) -> (x y x y)
								current->left_up_x = YOLO_MAX((pred_x - 0.5f * (pred_w - 1.0f)), 0.0f);
								current->left_up_y = YOLO_MAX((pred_y - 0.5f * (pred_h - 1.0f)), 0.0f);
								current->right_down_x = YOLO_MIN((pred_x + 0.5f * (pred_w - 1.0f)), IMAGE_W);
								current->right_down_y = YOLO_MIN((pred_y + 0.5f * (pred_h - 1.0f)), IMAGE_H);
								
								current->class_index = output_index - 5; // 类别索引，减 5 是因为前五个数据是 bbox + confidence 输出.
								current->score = class_confidence; // 置信度
								current->next = NULL;
								resltu_num ++;
								if (*output_result == NULL){ // 存储结果
									*output_result = current;
									former = current;
								
								}else{
									former->next = current;
									former = former->next;
								}
								current = NULL;
							}
						}
					}
				}
		}
    }
	return resltu_num;
}

void Yolo_Result_Sort(Yolov5_Result *output_result){ // 目前用这个做排序
	
	Yolov5_Result *comparable_node = NULL; // 右节点，挨个指向右边所有节点
	Yolov5_Result *comparable_next_node = NULL;
	Yolov5_Result *current_node = output_result; // 左节点，其与右边每个节点做比较
	Yolov5_Result *current_next_node = NULL;
	Yolov5_Result temp_node = {0};
	
	while (current_node != NULL){
	
		comparable_node = current_node->next;
		current_next_node = current_node->next; // 记录后续节点，方便调换数据后维持链表完整
	
		while (comparable_node != NULL){
			
			comparable_next_node = comparable_node->next; // 记录后续节点，方便调换数据后维持链表完整
			if (current_node->score >= comparable_node->score){ // 如果大于它，说明后面的比它小，比较下一个
				comparable_node = comparable_node->next;
				
			}else{
				// 当大于 current_confidence 时，数据做调换，内存不变，小的放后面去
				memcpy(&temp_node, current_node, sizeof(Yolov5_Result));
				memcpy(current_node, comparable_node, sizeof(Yolov5_Result));
				memcpy(comparable_node, &temp_node, sizeof(Yolov5_Result));
				current_node->next = current_next_node; // 链表接好
				comparable_node->next = comparable_next_node;
				comparable_node = comparable_node->next; //更新位置，因为当前节点已经小于current_node ，不必再做比较
			}
			
		}
		
		current_node = current_node->next;
	}
}

void Yolo_NMS(Yolov5_Result *output_result, float iou_threshold){ 

	Yolov5_Result *comparable_node = NULL; // 右节点，挨个指向右边所有节点
	Yolov5_Result *comparable_former_node = NULL;
	Yolov5_Result *current_node = output_result; // 左节点，其与右边每个节点做比较
	Yolov5_Result *temp_node = NULL;

	float overlap_left_x = 0.0f;
	float overlap_left_y = 0.0f;
	float overlap_right_x = 0.0f;
	float overlap_right_y = 0.0f;
	float current_area = 0.0f, comparable_area = 0.0f, overlap_area = 0.0f;
	float nms_ratio = 0.0f;
	float overlap_w = 0.0f, overlap_h = 0.0f;
	
	// yolo v5 的 nms 实现很优雅，我没在这里用，我在博客里介绍一下
	while (current_node != NULL){
	
		comparable_node = current_node->next;

		comparable_former_node = current_node;
		//printf("current_node->score = %f\n", current_node->score);
		current_area = (current_node->right_down_x - current_node->left_up_x) * (current_node->right_down_y - current_node->left_up_y);
	
		while (comparable_node != NULL){
			if (current_node->class_index != comparable_node->class_index){ // 如果类别不一致，没必要做 nms
				comparable_former_node = comparable_node;
				comparable_node = comparable_node->next;
				continue;
			}
			//printf("comparable_node->score = %f\n", comparable_node->score);
			comparable_area = (comparable_node->right_down_x - comparable_node->left_up_x) * (comparable_node->right_down_y - comparable_node->left_up_y);
			overlap_left_x = YOLO_MAX(current_node->left_up_x, comparable_node->left_up_x);
			overlap_left_y = YOLO_MAX(current_node->left_up_y, comparable_node->left_up_y);
			overlap_right_x = YOLO_MIN(current_node->right_down_x, comparable_node->right_down_x);
			overlap_right_y = YOLO_MIN(current_node->right_down_y, comparable_node->right_down_y);
			overlap_w = YOLO_MAX((overlap_right_x - overlap_left_x), 0.0F);
			overlap_h = YOLO_MAX((overlap_right_y - overlap_left_y), 0.0F);
			overlap_area = YOLO_MAX((overlap_w * overlap_h), 0.0f); // 重叠区域面积
			nms_ratio = overlap_area / (current_area + comparable_area - overlap_area);
	
			if (nms_ratio > iou_threshold){ // 重叠过大，去掉
				temp_node = comparable_node;
				comparable_node = comparable_node->next;
				comparable_former_node->next = comparable_node; // 链表接好
				free(temp_node);
			}else{
				comparable_former_node = comparable_node;
				comparable_node = comparable_node->next;
			}
			
		}
		//printf("loop end \n");
		current_node = current_node->next;
	}
}

/**
 * @brief output result and release 
 * 
 * @param temp 
 */

void printf_result(Yolov5_Result *temp){
	printf("--------------------\n");

	while (temp != NULL){
		
		printf("output_result->left_up_x = %f\t", temp->left_up_x);
		printf("output_result->left_up_y = %f\n", temp->left_up_y);

		printf("output_result->right_down_x = %f\t", temp->right_down_x);
		printf("output_result->right_down_y = %f\n", temp->right_down_y);

		printf("output_result->class_index = %d\t", temp->class_index);
		printf("output_result->score = %f\n\n", temp->score);
		
		temp = temp->next;
	}
	printf("--------------------\n");
}

void release_result(Yolov5_Result *output_result){

	Yolov5_Result *temp = NULL;

	while (output_result != NULL){
		
		temp = output_result;
		
		output_result = output_result->next;
	
		free(temp);
	}
}

/**
 * @brief Yolov5 sample
 * 
 */
void SAMPLE_SVP_NNIE_Yolov5(void)
{
    HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/dog_bike_car_448x448.bgr";
    HI_CHAR *pcModelName = "./data/nnie_model/detection/yolov5s6_448_voc_u8_rgb_inst.wk";
    HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
    Yolov5_Result *output_result = NULL;
    float confidence_threshold = 0.5f;

    /*Set configuration parameter*/
    f32PrintResultThresh = 0.2f;
    stNnieCfg.pszPic= pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*Yolov5 Load model*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stYolov5Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*Yolov5 parameter initialization*/
    /*Yolov5 software parameters are set in SAMPLE_SVP_NNIE_Yolov3_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov3_SoftwareInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 parameter initialization!\n");
    s_stYolov5NnieParam.pstModel = &s_stYolov5Model.stModel;
   
   // Height 25  Weight 3136  56*56
    SAMPLE_SVP_TRACE_INFO("The s_stYolov5Model00 Height is %d\n", s_stYolov5Model.stModel.astSeg->astDstNode->unShape.stWhc.u32Height);

    s32Ret = SAMPLE_SVP_NNIE_Yolov5_ParamInit(&stNnieCfg,&s_stYolov5NnieParam,&s_stYolov5SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov5_ParamInit failed!\n");


    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stYolov5NnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*NNIE process(process the 0-th segment)*/
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov5NnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    SAMPLE_SVP_TRACE_INFO("Forward!\n");

    /*Software process*/
    /*if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov3_GetResult
     function input datas are correct*/
    s32Ret = Yolov5_GetResult(&s_stYolov5NnieParam,strides, anchor_grids, map_size, &output_result, confidence_threshold);
    // s32Ret = SAMPLE_SVP_NNIE_Yolov3_GetResult(&s_stYolov5NnieParam,&s_stYolov5SoftwareParam);
    // SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //     "Error,SAMPLE_SVP_NNIE_Yolov5_GetResult failed!\n");

    if (output_result != NULL)
    {
        Yolo_Result_Sort(output_result);
        Yolo_NMS(output_result,f32PrintResultThresh);
    }
    

     /*print result, this sample has 81 classes:
      class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike      class 5:aeroplane
      class 6:bus             class 7:train        class 8:truck           class 9:boat           class 10:traffic light
      class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class 15:bird
      class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow
      class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack
      class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee
      class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat
      class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle
      class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon
      class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange
      class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut
      class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed
      class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse
      class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven
      class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock
      class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 result:\n");
    printf_result(output_result); // 打印结果看下，这里打印的已经是可以输出画框的结果了
	
	release_result(output_result); // 释放内存，一帧结束
    // (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov3SoftwareParam.stDstScore,
    //     &s_stYolov3SoftwareParam.stDstRoi, &s_stYolov3SoftwareParam.stClassRoiNum,f32PrintResultThresh);


YOLOV5_FAIL_0:
    SAMPLE_SVP_NNIE_Yolov5_Deinit(&s_stYolov5NnieParam,&s_stYolov5SoftwareParam,&s_stYolov5Model);
    SAMPLE_COMM_SVP_CheckSysExit();
}


/******************************************************************************
* function : Yolov5 Procession ViToVo
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov5_Proc_ViToVo(SAMPLE_SVP_NNIE_PARAM_S *pstParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSwParam, VIDEO_FRAME_INFO_S* pstExtFrmInfo,
    HI_U32 u32BaseWidth,HI_U32 u32BaseHeight)
{
    HI_S32 s32Ret = HI_FAILURE;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
    Yolov5_Result *output_result = NULL;
    float confidence_threshold = 0.5f;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    HI_FLOAT f32PrintResultThresh = 0.01f;
    /*SP420*/
    pstParam->astSegData[stInputDataIdx.u32SegIdx].astSrc[stInputDataIdx.u32NodeIdx].u64VirAddr = pstExtFrmInfo->stVFrame.u64VirAddr[0];
    pstParam->astSegData[stInputDataIdx.u32SegIdx].astSrc[stInputDataIdx.u32NodeIdx].u64PhyAddr = pstExtFrmInfo->stVFrame.u64PhyAddr[0];
    pstParam->astSegData[stInputDataIdx.u32SegIdx].astSrc[stInputDataIdx.u32NodeIdx].u32Stride  = pstExtFrmInfo->stVFrame.u32Stride[0];
    
    SAMPLE_SVP_TRACE_INFO("The s_stYolov5ModelFpst Width is %d\n", pstExtFrmInfo->stVFrame.u32Width);
    SAMPLE_SVP_TRACE_INFO("The s_stYolov5ModelFpst Height is %d\n", pstExtFrmInfo->stVFrame.u32Height);


    // 与Rfcn不同YOLOv3是调用 SAMPLE_SVP_NNIE_Forward 进行解析
    //s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov3NnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);

    //yolov5 forward
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov5NnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    // SAMPLE_SVP_TRACE_INFO("Yolov5 result:\n");
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov5_Proc failed!\n");
    /*draw result, this sample has 21 classes:
     class 0:background     class 1:plane           class 2:bicycle
     class 3:bird           class 4:boat            class 5:bottle
     class 6:bus            class 7:car             class 8:cat
     class 9:chair          class10:cow             class11:diningtable
     class 12:dog           class13:horse           class14:motorbike
     class 15:person        class16:pottedplant     class17:sheep
     class 18:sofa          class19:train           class20:tvmonitor*/
    /*Software process*/
    /*if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov3_GetResult
     function input datas are correct*/

    //yolov5 GetResult
    s32Ret = Yolov5_GetResult(&s_stYolov5NnieParam,strides, anchor_grids, map_size, &output_result, confidence_threshold);

    // s32Ret = SAMPLE_SVP_NNIE_Yolov3_GetResult(&s_stYolov3NnieParam,&s_stYolov3SoftwareParam);
 
    //  (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&(pstSwParam->stDstScore), &(pstSwParam->stDstRoi), &(pstSwParam->stClassRoiNum),f32PrintResultThresh);
 
    //  SAMPLE_SVP_TRACE_INFO("Yolov3 result:\n");
    // (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov3SoftwareParam.stDstScore,
    //    &s_stYolov3SoftwareParam.stDstRoi, &s_stYolov3SoftwareParam.stClassRoiNum,f32PrintResultThresh);

    if (output_result != NULL)
    {
        Yolo_Result_Sort(output_result);
        Yolo_NMS(output_result,f32PrintResultThresh);
    }
    
    SAMPLE_SVP_TRACE_INFO("Yolov5 result:\n");
    printf_result(output_result); // 打印结果看下，这里打印的已经是可以输出画框的结果了
	
	release_result(output_result); // 释放内存，一帧结束
    
    // SAMPLE_SVP_TRACE_INFO("The Print u32ClassNum is %d\n",s_stYolov3SoftwareParam.u32ClassNum);
    // SAMPLE_SVP_TRACE_INFO("The Print u32NmsThresh is %d\n",s_stYolov3SoftwareParam.u32NmsThresh);
    // SAMPLE_SVP_TRACE_INFO("The Print Width is %d\n",s_stYolov3SoftwareParam.stDstScore.unShape.stWhc.u32Width);
	
    // wait handle  draw
    // s32Ret = SAMPLE_SVP_NNIE_RoiToRect_Yolov3(&(pstSwParam->stDstScore),
    // &(pstSwParam->stDstRoi), &(pstSwParam->stClassRoiNum), pstSwParam->af32ScoreThr,HI_TRUE,&(pstSwParam->stRect),
    // pstExtFrmInfo->stVFrame.u32Width, pstExtFrmInfo->stVFrame.u32Height,u32BaseWidth,u32BaseHeight);
    // SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //     "Error(%#x),SAMPLE_SVP_NNIE_RoiToRect failed!\n",s32Ret); 
    return s32Ret;
}

/******************************************************************************
* function : Yolov5 vi to vo thread entry
******************************************************************************/
static HI_VOID* SAMPLE_SVP_NNIE_Yolov5_ViToVo_thread(HI_VOID* pArgs)
{
    HI_S32 s32Ret;
    SAMPLE_SVP_NNIE_PARAM_S *pstParam;
    // use yolov3 software_param
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSwParam;
    VIDEO_FRAME_INFO_S stBaseFrmInfo;
    VIDEO_FRAME_INFO_S stExtFrmInfo;
    HI_S32 s32MilliSec = 20000;
    VO_LAYER voLayer = 0;
    VO_CHN voChn = 0;
    HI_S32 s32VpssGrp = 0;
    HI_S32 as32VpssChn[] = {VPSS_CHN0, VPSS_CHN1};

    pstParam = &s_stYolov5NnieParam;
    pstSwParam = &s_stYolov5SoftwareParam;

    while (HI_FALSE == s_bNnieStopSignal)
    {
        s32Ret = HI_MPI_VPSS_GetChnFrame(s32VpssGrp, as32VpssChn[1], &stExtFrmInfo, s32MilliSec);
        if(HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_GetChnFrame failed, VPSS_GRP(%d), VPSS_CHN(%d)!\n",
                s32Ret,s32VpssGrp, as32VpssChn[1]);
            continue;
        }
        s32Ret = HI_MPI_VPSS_GetChnFrame(s32VpssGrp, as32VpssChn[0], &stBaseFrmInfo, s32MilliSec);
        // printf("stBaseFrmInfo Height:%d\n",stBaseFrmInfo.stVFrame.u32Height);
        // printf("stBaseFrmInfo width:%d\n",stBaseFrmInfo.stVFrame.u32Width);
        SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS!=s32Ret, EXT_RELEASE,
            "Error(%#x),HI_MPI_VPSS_GetChnFrame failed, VPSS_GRP(%d), VPSS_CHN(%d)!\n",
            s32Ret,s32VpssGrp, as32VpssChn[0]);
        
        // SAMPLE_SVP_TRACE_INFO("The Width is %d\n", stExtFrmInfo.stVFrame.u32Width);  416
	    // SAMPLE_SVP_TRACE_INFO("The Height is %d\n", stExtFrmInfo.stVFrame.u32Height);
        // 存在疑问 stBaseFrmInfo.stVFrame.u32Height
        s32Ret = SAMPLE_SVP_NNIE_Yolov5_Proc_ViToVo(pstParam,pstSwParam, &stExtFrmInfo,
        stBaseFrmInfo.stVFrame.u32Width,stBaseFrmInfo.stVFrame.u32Height);
        SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS!=s32Ret, BASE_RELEASE,
            "Error(%#x),SAMPLE_SVP_NNIE_YOLOV5_Proc failed!\n", s32Ret);
        SAMPLE_SVP_TRACE_INFO("The Width is %d\n", stExtFrmInfo.stVFrame.u32Width);
	    SAMPLE_SVP_TRACE_INFO("The Height is %d\n", stExtFrmInfo.stVFrame.u32Height);
	 
        //Draw rect
        // s32Ret = SAMPLE_COMM_SVP_NNIE_FillRect(&stBaseFrmInfo, &(pstSwParam->stRect), 0x0000FF00);
        // SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS!=s32Ret, BASE_RELEASE,
        //     "SAMPLE_COMM_SVP_NNIE_FillRect failed, Error(%#x)!\n", s32Ret);
        // s32Ret = HI_MPI_VO_SendFrame(voLayer, voChn, &stBaseFrmInfo, s32MilliSec);
        // SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS!=s32Ret, BASE_RELEASE,
        //     "HI_MPI_VO_SendFrame failed, Error(%#x)!\n", s32Ret);
        BASE_RELEASE:
            s32Ret = HI_MPI_VPSS_ReleaseChnFrame(s32VpssGrp,as32VpssChn[0], &stBaseFrmInfo);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                    s32Ret,s32VpssGrp,as32VpssChn[0]);
            }
        EXT_RELEASE:
            s32Ret = HI_MPI_VPSS_ReleaseChnFrame(s32VpssGrp,as32VpssChn[1], &stExtFrmInfo);
            if (HI_SUCCESS != s32Ret)
            {
                SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                    s32Ret,s32VpssGrp,as32VpssChn[1]);
            }
    }
    return HI_NULL;
}


/******************************************************************************
* function : Yolov5 vi to vo real time detection
******************************************************************************/
void SAMPLE_SVP_NNIE_Yolov5_Vivo(void)
{
 //   HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/dog_bike_car_416x416.bgr";   
    HI_CHAR *pcModelName = "./data/nnie_model/detection/yolov5s6_448_voc_YVU420_rgb_inst.wk";
    // HI_CHAR *pcModelName = "./data/nnie_model/detection/inst_yolov3_cycle.wk";
    SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};
    SIZE_S stSize;
    PIC_SIZE_E enSize = PIC_CIF;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_CHAR acThreadName[16] = {0};
    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();
    /******************************************
     step 1: start vi vpss vo
     ******************************************/
    s_stYolov5Switch.bVenc = HI_FALSE;
    s_stYolov5Switch.bVo   = HI_TRUE;
    s32Ret = SAMPLE_COMM_IVE_StartViVpssVencVo(&s_stViConfig,&s_stYolov5Switch,&enSize);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV5_FAIL_1,
        "Error(%#x),SAMPLE_COMM_IVE_StartViVpssVencVo failed!\n", s32Ret);
    s32Ret = SAMPLE_COMM_SYS_GetPicSize(enSize, &stSize);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV5_FAIL_1,
        "Error(%#x),SAMPLE_COMM_SYS_GetPicSize failed!\n", s32Ret);

    SAMPLE_SVP_TRACE_INFO("The  stSize Width is %d\n", stSize.u32Width);  
	SAMPLE_SVP_TRACE_INFO("The  stSize Height is %d\n", stSize.u32Height);
    stSize.u32Width = 448;
    stSize.u32Height = 448;
    /******************************************
     step 2: init NNIE param
     ******************************************/
    stNnieCfg.pszPic= NULL;
    stNnieCfg.u32MaxInputNum = 1;
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core
    /*Yolov5 Load model*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stYolov5Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
    /*Yolov3 parameter initialization*/
    /*Yolov3 software parameters are set in SAMPLE_SVP_NNIE_Yolov3_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov3_SoftwareInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 parameter initialization!\n");
    s_stYolov5NnieParam.pstModel = &s_stYolov5Model.stModel;

    SAMPLE_SVP_TRACE_INFO("The s_stYolov5Model  real time Height is %d\n", s_stYolov5Model.stModel.astSeg->astDstNode->unShape.stWhc.u32Height);

    s32Ret = SAMPLE_SVP_NNIE_Yolov5_ParamInit(&stNnieCfg,&s_stYolov5NnieParam,&s_stYolov5SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,YOLOV5_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov5_ParamInit failed!\n");
    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Yolov5 start!\n");
    s_bNnieStopSignal = HI_FALSE;
    /******************************************
      step 3: Create work thread
     ******************************************/
    snprintf(acThreadName, 16, "NNIE_ViToVo");
    prctl(PR_SET_NAME, (unsigned long)acThreadName, 0,0,0);
    pthread_create(&s_hNnieThread, 0, SAMPLE_SVP_NNIE_Yolov5_ViToVo_thread, NULL);
     SAMPLE_PAUSE();
    s_bNnieStopSignal = HI_TRUE;
    pthread_join(s_hNnieThread, HI_NULL);
    s_hNnieThread = 0;
YOLOV5_FAIL_1:
    SAMPLE_SVP_NNIE_Yolov5_Deinit(&s_stYolov5NnieParam,&s_stYolov5SoftwareParam,&s_stYolov5Model);
YOLOV5_FAIL_0:
    SAMPLE_COMM_IVE_StopViVpssVencVo(&s_stViConfig,&s_stYolov5Switch);
}

/******************************************************************************
* function : Yolov5 sample signal handle
******************************************************************************/
void SAMPLE_SVP_NNIE_Yolov5_Vivo_HandleSig(void)
{
    s_bNnieStopSignal = HI_TRUE;
    if (0 != s_hNnieThread)
    {
        pthread_join(s_hNnieThread, HI_NULL);
        s_hNnieThread = 0;
    }
    SAMPLE_SVP_NNIE_Yolov5_Deinit(&s_stYolov5NnieParam,&s_stYolov5SoftwareParam,&s_stYolov5Model);
    memset(&s_stYolov5NnieParam,0,sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    memset(&s_stYolov5SoftwareParam,0,sizeof(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S));
    memset(&s_stYolov5Model,0,sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_IVE_StopViVpssVencVo(&s_stViConfig,&s_stYolov5Switch);
}
