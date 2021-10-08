EXEC = main

$(EXEC):
    
#OUROBOROS
	#nvcc -G -g -arch=sm_75 --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 --expt-relaxed-constexpr -I include -I Ouroboros_origin/include main.cu

# HALLOC:
	nvcc -G -g --resource-usage -Xptxas --warn-on-spills --maxrregcount 32 --expt-relaxed-constexpr -L cuda.h -I GPUMemManSurvey/include -I GPUMemManSurvey/frameworks/halloc/repository/src/ -I GPUMemManSurvey/frameworks/ouroboros/repository/include -I include -I GPUMemManSurvey/frameworks/halloc main.cu

    #nvcc -G -g --resource-usage -Xptxas --warn-on-spills --expt-relaxed-constexpr -L cuda.h -I include/Ouroboros/include main.cu


    
