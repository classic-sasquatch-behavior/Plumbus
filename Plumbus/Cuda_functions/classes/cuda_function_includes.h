#pragma once


#include"../headers/AP/dampen_messages.cuh"
#include"../headers/AP/extract_exemplars.cuh"
#include"../headers/AP/form_similarity_matrix.cuh"
#include"../headers/AP/form_similarity_matrix_color.cuh"
#include"../headers/AP/presort_matrix_ap.cuh"
#include"../headers/AP/update_availibility_matrix.cuh"
#include"../headers/AP/update_critereon_matrix.cuh"
#include"../headers/AP/update_responsibility_matrix.cuh"

#include"../headers/generic operations/exclusive_scan_vec.cuh"
#include"../headers/generic operations/pop_elm_vec.cuh"
#include"../headers/generic operations/sum_vec.cuh"

#include"../headers/misc/fast_selective_blur.cuh"
#include"../headers/misc/matrix_operations.cuh"
#include"../headers/misc/selective_blur.h"

#include"../headers/SLIC/find_labels.cuh"
#include"../headers/SLIC/update_centers.cuh"
#include"../headers/SLIC/absorb_small_blobs.cuh"
#include"../headers/SLIC/produce_ordered_labels.cuh"
#include"../headers/SLIC/separate_blobs.cuh"

#include"../headers/SP processing/find_borders.cuh"





