CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#Change PROJECT_NAME to the name of your project
PROJECT(fuzzyClustering)

#include some macros from another file...
INCLUDE(${CMAKE_SOURCE_DIR}/IJMacros.txt)

#The following lines are required to use Dart
ENABLE_TESTING()
INCLUDE(Dart)
FIND_PROGRAM(ITK_TEST_DRIVER itkTestDriver)
SET(TEST_COMMAND ${ITK_TEST_DRIVER} --add-before-env PATH ${CMAKE_CURRENT_BINARY_DIR})


#Declare any external dependencies that your project may have here.
#examples include: ITK, VTK, JPEG, PNG, OpenGL, ZLIB, Perl, Java
#If you're not sure what name to use, look in the Modules directory of your
#cmake install and check that a file named Find(Package).cmake exists
SET(Required_Packages
 ITK
)

#this foreach loads all of the packages that you specified as required.
#It shouldn't need to be modified.
FOREACH(Package ${Required_Packages})
  LOADPACKAGE(${Package})
ENDFOREACH(Package)

INCLUDE(${ITK_USE_FILE})

#Set any libraries that your project depends on.
#examples: ITKCommon, VTKRendering, etc
IF( "${ITK_VERSION_MAJOR}" LESS 4 )
	SET(Libraries
 		ITKCommon
 		ITKIO
		ITKStatistics
		)
ELSE( "${ITK_VERSION_MAJOR}" LESS 4 )
	SET(Libraries
 	${ITK_LIBRARIES}
 	)
ENDIF( "${ITK_VERSION_MAJOR}" LESS 4 )

IF(BUILD_TESTING)
	FOREACH(CurrentExe "FCMClassification2D" "KFCMSClassification2D" "MSKFCMClassification2D"
	"MSKFCMOpening2D" "MSKFCMClassification3D")
		ADD_EXECUTABLE(${CurrentExe} ${CurrentExe}.cxx)
		TARGET_LINK_LIBRARIES(${CurrentExe} ${Libraries})
	ENDFOREACH(CurrentExe)
ENDIF(BUILD_TESTING)

#the following line is an example of how to add a test to your project.
#Testname is the title for this particular test.  ExecutableToRun is the
#program which will be running this test.  It can either be a part of this
#project or an external executable.  After that list any args that are needed
#for this test.  Include as many tests as you like.  If your project doesn't have
#any tests you can comment out or delete the following line.


SET(DIR_INPUT_IM ${CMAKE_CURRENT_SOURCE_DIR}/Images/input)
SET(DIR_OUTPUT_IM ${CMAKE_CURRENT_SOURCE_DIR}/Images/output)

SET(centroidVal1 -940 -900 -840 -720 -500 -240)
SET(centroidVal2 -450 -400 -350 -650 -850 -250)
SET(nThreads 4 3 2 1 4 4)
SET(numIm 1 2 3 4 5 6)

FOREACH(i RANGE 0 5)
	LIST(GET centroidVal1 ${i} centr1)
	LIST(GET centroidVal2 ${i} centr2)
	LIST(GET nThreads ${i} nThrd)
	LIST(GET numIm ${i} nIm)

	ADD_TEST(FCMClassification2D_${nIm} ${TEST_COMMAND}
   		FCMClassification2D ${DIR_INPUT_IM}/inputIm${nIm}.dcm outIm${nIm}.dcm 500 0.001 2 
   		${nThrd} 2 ${centr1} ${centr2} -f -2000 --compare 
   		outIm${nIm}.dcm ${DIR_OUTPUT_IM}/FCM/outIm${nIm}.dcm 
   		--compareIntensityTolerance 0
   		--compareRadiusTolerance 0
   		--compareNumberOfPixelsTolerance 0
   	)
ENDFOREACH(i)

SET(sigma 750 650 850 550 450 450)
SET(m 1.5 1.4 1.4 2 2 2)
SET(alpha 1 2 1 1 1 1)

FOREACH(i RANGE 0 5)
	LIST(GET centroidVal1 ${i} centr1)
	LIST(GET centroidVal2 ${i} centr2)
	LIST(GET numIm ${i} nIm)
	LIST(GET sigma ${i} sig)
	LIST(GET m ${i} mVal)
	LIST(GET alpha ${i} alph)

	ADD_TEST(KFCMSClassification2D_${nIm} ${TEST_COMMAND}
   		KFCMSClassification2D ${DIR_INPUT_IM}/inputIm${nIm}.dcm outIm${nIm}.dcm 
   		500 0.001 ${mVal} ${alph} 4 2 ${centr1} ${centr2} ${sig} 1 1 -f -2000 --compare
   		outIm${nIm}.dcm ${DIR_OUTPUT_IM}/KFCMS/outIm${nIm}.dcm
   		--compareIntensityTolerance 0
   		--compareRadiusTolerance 0
   		--compareNumberOfPixelsTolerance 0
   	)
ENDFOREACH(i)

SET(sigma 450 350 400 450 450 200)
SET(radius 3 3 5 7 1 1)

FOREACH(i RANGE 0 5)
	LIST(GET centroidVal1 ${i} centr1)
	LIST(GET centroidVal2 ${i} centr2)
	LIST(GET numIm ${i} nIm)
	LIST(GET sigma ${i} sig)
	LIST(GET radius ${i} rad)

	ADD_TEST(MSKFCMClassification2D_${nIm} ${TEST_COMMAND}
   		MSKFCMClassification2D ${DIR_INPUT_IM}/inputIm${nIm}.dcm outIm${nIm}.dcm 
   		500 0.001 2 2 1 4 2 ${centr1} ${centr2} ${sig} ${rad} ${rad} -f -2000 --compare
   		outIm${nIm}.dcm ${DIR_OUTPUT_IM}/MSKFCM/outIm${nIm}.dcm 
   		--compareIntensityTolerance 0
   		--compareRadiusTolerance 0
   		--compareNumberOfPixelsTolerance 0
   	)
ENDFOREACH(i)

ADD_TEST(MSKFCMOpening2D ${TEST_COMMAND}
	MSKFCMOpening2D ${DIR_INPUT_IM}/inputIm1.dcm outIm1.dcm 
	500 0.001 2 2 1 25 0 2 4 3 -1000 -940 -450 450 3 3 --compare
	outIm1.dcm ${DIR_OUTPUT_IM}/MSKFCMOpening/outIm1.dcm 
	--compareIntensityTolerance 0
	--compareRadiusTolerance 0
	--compareNumberOfPixelsTolerance 0
)

ADD_TEST(MSKFCMClassification3D ${TEST_COMMAND}
	MSKFCMClassification3D ${DIR_INPUT_IM}/serie ./ 
	500 0.001 2 2 1 4 2 -940 -450 450 1 1 1 -f -2000 
	--compare 000001.dcm ${DIR_OUTPUT_IM}/Serie/000001.dcm 
	--compare 000002.dcm ${DIR_OUTPUT_IM}/Serie/000002.dcm
	--compare 000003.dcm ${DIR_OUTPUT_IM}/Serie/000003.dcm
	--compare 000004.dcm ${DIR_OUTPUT_IM}/Serie/000004.dcm
	--compare 000005.dcm ${DIR_OUTPUT_IM}/Serie/000005.dcm
	--compare 000006.dcm ${DIR_OUTPUT_IM}/Serie/000006.dcm
	--compare 000007.dcm ${DIR_OUTPUT_IM}/Serie/000007.dcm
	--compare 000008.dcm ${DIR_OUTPUT_IM}/Serie/000008.dcm
	--compare 000009.dcm ${DIR_OUTPUT_IM}/Serie/000009.dcm
	--compare 000010.dcm ${DIR_OUTPUT_IM}/Serie/000010.dcm
	--compareIntensityTolerance 0
	--compareRadiusTolerance 0
	--compareNumberOfPixelsTolerance 0
)
