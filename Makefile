EXEC = main

$(EXEC) : 
	#nvcc --expt-relaxed-constexpr -L cuda.h error_utils.cu -I include/Ouroboros/include error_utils.cu main.cu
	nvcc -G -g --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 --expt-relaxed-constexpr \
    -L cuda.h -I GPUMemManSurvey/include -I GPUMemManSurvey/frameworks/halloc/repository/src/ \
    -I GPUMemManSurvey/frameworks/ouroboros/repository/include \
    -I include -I GPUMemManSurvey/frameworks/halloc main.cu
    

    #-L cuda.h -I Ouroboros_origin/include -I GPUMemManSurvey/frameworks/halloc/repository/src/ \

	#nvcc -G -g -arch=sm_75 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 --expt-relaxed-constexpr -L cuda.h -I Ouroboros_origin/include -I halloc -I include main.cu
	#nvcc -G -g --resource-usage -Xptxas --warn-on-spills --expt-relaxed-constexpr -L cuda.h -I include/Ouroboros/include main.cu
