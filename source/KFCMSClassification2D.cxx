#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSimpleFilterWatcher.h"

#include "itkKFCMSClassifierInitializationImageFilter.h"
#include "itkFuzzyClassifierImageFilter.h"

int
main(int argc, char * argv[])
{

  if (argc < 10)
    {
      std::cerr << "usage: " << argv[0]
          << " input output nmaxIter error m alpha"
            "numThreads numClasses { centroids_1,...,centroid_numClusters }"
            "sigma radius [ -f valBackground ]" << std::endl;
      exit(1);
    }

  const int dim = 2;
  typedef signed short IPixelType;

  typedef unsigned char OPixelType;

  typedef itk::Image<IPixelType, dim> IType;
  typedef itk::Image<OPixelType, dim> OType;

  typedef itk::ImageFileReader<IType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);

  typedef itk::FuzzyClassifierInitializationImageFilter<IType>
      TFuzzyClassifier2D;
  typedef itk::KFCMSClassifierInitializationImageFilter<IType> TClassifierKFCMS;

  TClassifierKFCMS::Pointer classifier = TClassifierKFCMS::New();
  itk::SimpleFilterWatcher watcher(classifier, "KFCMS classifier");
  classifier->SetMaximumNumberOfIterations(atoi(argv[3]));
  classifier->SetMaximumError(atof(argv[4]));
  classifier->SetM(atof(argv[5]));
  std::cout<<atof(argv[5])<<std::endl;
  classifier->SetAlpha(atof(argv[6]));
  classifier->SetNumberOfThreads(atoi(argv[7]));

  int numClasses = atoi(argv[8]);
  classifier->SetNumberOfClasses(numClasses);

  TFuzzyClassifier2D::CentroidArrayType centroidsArray;

  int argvIndex = 9;
  for (int i = 0; i < numClasses; i++)
    {
      centroidsArray.push_back(atof(argv[argvIndex]));
      ++argvIndex;
    }

  classifier->SetCentroids(centroidsArray);

  typedef TFuzzyClassifier2D::CentroidType TCentroid;
  typedef itk::Statistics::RBFKernelInducedDistanceMetric<TCentroid>
      RBFKernelType;
  RBFKernelType::Pointer kernelDistancePtr = RBFKernelType::New();
  kernelDistancePtr->SetA(2.0);
  kernelDistancePtr->SetB(1.0);
  kernelDistancePtr->SetSigma(atoi(argv[argvIndex]));
  classifier->SetKernelDistanceMetric(static_cast<
      TClassifierKFCMS::KernelDistanceMetricPointer >( kernelDistancePtr));

  typedef itk::FlatStructuringElement<
      dim> StructuringElement2DType;
  StructuringElement2DType::RadiusType elementRadius;
  for (int i = 0; i < dim; i++)
    {
    ++argvIndex;
    elementRadius[i] = atoi(argv[argvIndex]);
    }
  StructuringElement2DType structuringElement = StructuringElement2DType::Box(
      elementRadius);
  classifier->SetStructuringElement(structuringElement);

  if ( (argc-1 > argvIndex ) && (strcmp(argv[argvIndex + 1], "-f") == 0) )
    {
      classifier->SetIgnoreBackgroundPixels(true);
      classifier->SetBackgroundPixel(atof(argv[argvIndex + 2]));
    }

  classifier->SetInput(reader->GetOutput());

  typedef itk::FuzzyClassifierImageFilter<TClassifierKFCMS::OutputImageType>
      TLabelClassifier2D;
  TLabelClassifier2D::Pointer labelClass = TLabelClassifier2D::New();
  labelClass->SetInput(classifier->GetOutput());

  typedef itk::ImageFileWriter<OType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(labelClass->GetOutput());
  writer->SetFileName(argv[2]);

  try
    {
    writer->Update();
    }
  catch (itk::ExceptionObject & excp)
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
