#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkFCMClassifierInitializationImageFilter.h"
#include "itkFuzzyClassifierImageFilter.h"

int
main(int argc, char * argv[])
{

  if (argc < 7)
    {
      std::cerr << "usage: " << argv[0] << " input output nmaxIter error m "
        "numThreads numClasses { centroids_1,...,centroid_numClusters } "
        "[ -f valBackground ]" << std::endl;
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

  typedef itk::FCMClassifierInitializationImageFilter<IType> TClassifierFCM;

  TClassifierFCM::Pointer classifier = TClassifierFCM::New();
  classifier->SetMaximumNumberOfIterations(atoi(argv[3]));
  classifier->SetMaximumError(atof(argv[4]));
  classifier->SetM(atof(argv[5]));
  classifier->SetNumberOfThreads(atoi(argv[6]));

  int numClasses = atoi(argv[7]);
  classifier->SetNumberOfClasses(numClasses);

  TFuzzyClassifier2D::CentroidArrayType centroidsArray;

  int argvIndex = 8;
  for (int i = 0; i < numClasses; i++)
    {
      centroidsArray.push_back(atof(argv[argvIndex]));
      ++argvIndex;
    }

  classifier->SetCentroids(centroidsArray);

  if ( (argc-1 > argvIndex ) && (strcmp(argv[argvIndex], "-f") == 0) )
    {
    classifier->SetIgnoreBackgroundPixels(true);
    classifier->SetBackgroundPixel(atof(argv[argvIndex + 1]));
    }

  classifier->SetInput(reader->GetOutput());

  typedef itk::FuzzyClassifierImageFilter<TClassifierFCM::OutputImageType>
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
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
