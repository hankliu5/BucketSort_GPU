extern "C"
{
int pthread_sort(int num_of_elements, float *data);
int cuda_sort(int num_of_elements, float *data, int step);
//int fpga_sort(int num_of_elements, float *data);
}