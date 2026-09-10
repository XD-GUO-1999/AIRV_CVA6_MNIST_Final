// N2D2 auto-generated file.
// @ Fri Sep 10 16:23:21 2021

#ifndef N2D2_EXPORTCPP_MEM_INFO_H
#define N2D2_EXPORTCPP_MEM_INFO_H

#define MEMORY_SIZE 2160 //the total memory size (in bytes) used for the network
#define MEMORY_ALIGNMENT 1 //the alignment of the memory used for the network, the method of N2D2
#define ENV_MEM_SIZE 1 //one byte per input pixel
#define ENV_MEM_STRIDE 1 //one byte per input pixel
#define ENV_MEM_LENGTH 24 //the number of lines of the input (height)
#define ENV_MEM_COUNT 24 //the number of columns of the input (width)
#define ENV_MEM_CONT_OFFSET 0 //no offset, that means the input is stored at the beginning of the memory
#define ENV_MEM_CONT_SIZE 576 // the size of the input in bytes (24*24*1)
#define ENV_MEM_WRAP_OFFSET 0 //no wrapping, that means the input is stored in a contiguous way in the memory
#define ENV_MEM_WRAP_SIZE 0 //no wrapping
#define CONV1_MEM_SIZE 16 //16 bytes per output pixel (16 channels, 1 byte per channel)
#define CONV1_MEM_STRIDE 16 // 16 byte per output pixel
#define CONV1_MEM_LENGTH 11 //the number of lines of the output (height)
#define CONV1_MEM_COUNT 11 //the number of columns of the output (width)
#define CONV1_MEM_CONT_OFFSET 576 //the offset of the conv1 output in the memory, that means the conv1 output is stored after the input in the memory (576)
#define CONV1_MEM_CONT_SIZE 1584 //the size of the conv1 output in bytes (11*11*16 - 352), because 576 + 1584 = 2160
#define CONV1_MEM_WRAP_OFFSET 0 //no wrapping offset, which means the conv1 output will be stored at the beginning of the memory after reaching the end of the contiguous area (576 + 1584 = 2160)
#define CONV1_MEM_WRAP_SIZE 352 // the size of the wrapping area for the conv1 output in bytes (352)
#define CONV2_MEM_SIZE 24 //24 bytes per output pixel (24 channels, 1 byte per channel)
#define CONV2_MEM_STRIDE 24 //24 byte per output pixel
#define CONV2_MEM_LENGTH 4 //the number of lines of the output (height)
#define CONV2_MEM_COUNT 4 //the number of columns of the output (width)
#define CONV2_MEM_CONT_OFFSET 352 //conv2 output is stored after the conv1 output in memory
#define CONV2_MEM_CONT_SIZE 384 //the size of the conv2 output in bytes (4*4*24)
#define CONV2_MEM_WRAP_OFFSET 0 //no wrapping
#define CONV2_MEM_WRAP_SIZE 0 //no wrapping
#define FC1_MEM_SIZE 150 //150 bytes per output pixel (150 channels, 1 byte per channel)
#define FC1_MEM_STRIDE 150 //150 bytes per output pixel (150 channels, 1 byte per channel)
#define FC1_MEM_LENGTH 1 //the number of lines of the output (height)
#define FC1_MEM_COUNT 1 //the number of columns of the output (width)
#define FC1_MEM_CONT_OFFSET 736 //fc1 output is stored after the conv2 output in memory(352 + 384)
#define FC1_MEM_CONT_SIZE 150 //the size of the fc1 output in bytes (1*1*150)
#define FC1_MEM_WRAP_OFFSET 0 //no wrapping
#define FC1_MEM_WRAP_SIZE 0 //no wrapping
#define FC2_MEM_SIZE 10 //10 bytes per output pixel (10 channels, 1 byte per channel)
#define FC2_MEM_STRIDE 10 //10 bytes per output pixel (10 channels, 1 byte per channel)
#define FC2_MEM_LENGTH 1 //the number of lines of the output (height)
#define FC2_MEM_COUNT 1 //the number of columns of the output (width)
#define FC2_MEM_CONT_OFFSET 886 //fc2 output is stored after the fc1 output in memory(736 + 150)
#define FC2_MEM_CONT_SIZE 10 //the size of the fc1 output in bytes (1*1*10)
#define FC2_MEM_WRAP_OFFSET 0 //no wrapping
#define FC2_MEM_WRAP_SIZE 0 //no wrapping

#endif
