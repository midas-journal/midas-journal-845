#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkSimpleFilterWatcher.h"

#include "itkFuzzyClassifierInitializationImageFilter.h"
#include "itkMSKFCMClassifierInitializationImageFilter.h"
#include "itkFuzzyClassifierImageFilter.h"

int
main(int argc, char * argv[])
{

  if (argc < 11)
    {
      std::cerr << "usage: " << argv[0] << " input output nmaxIter error m P Q"
        "numThreads numClasses { centroids_1,...,centroid_numClusters } sigma"
        "radius [ -f valBackground ]" << std::endl;
      exit(1);
    }

  const int dim = 3;
  typedef signed short IPixelType;

  typedef unsigned char OPixelType;

  typedef itk::Image<IPixelType, dim> IType;
  typedef itk::Image<OPixelType, dim> OType;

  typedef itk::Image<OPixelType, 2> OType2D;

  typedef itk::GDCMImageIO DicomIOType;

  typedef itk::ImageSeriesReader<IType> DicomReaderType;

  typedef itk::GDCMSeriesFileNames NamesGeneratorType;

  DicomReaderType::Pointer reader = DicomReaderType::New();
  DicomIOType::Pointer dicomIO = DicomIOType::New();
  reader->SetImageIO(dicomIO);
  NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();
  namesGenerator->SetUseSeriesDetails(true);
  namesGenerator->SetDirectory(argv[1]);

  typedef std::vector<std::string> IdsContainerType;
  IdsContainerType ids;
  try
    {
    ids = namesGenerator->GetSeriesUIDs();
    }
  catch (itk::ExceptionObject & excp)
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  std::string identifier = ids.begin()->c_str();

  typedef std::vector<std::string> NamesContainerType;
  NamesContainerType names = namesGenerator->GetFileNames(identifier);

  reader->SetFileNames(names);

  typedef itk::FuzzyClassifierInitializationImageFilter<IType>
      TFuzzyClassifier2D;

  typedef itk::MSKFCMClassifierInitializationImageFilter<IType>
      TClassifierMSKFCM;

  typedef itk::FuzzyClassifierInitializationImageFilter<IType>
      TFuzzyClassifier;

  TClassifierMSKFCM::Pointer classifier = TClassifierMSKFCM::New();
  classifier->SetMaximumNumberOfIterations(atoi(argv[3]));
  classifier->SetMaximumError(atof(argv[4]));
  classifier->SetM(atoi(argv[5]));
  classifier->SetP(atof(argv[6]));
  classifier->SetQ(atof(argv[7]));
  classifier->SetNumberOfThreads(atoi(argv[8]));

  int numClasses = atoi(argv[9]);
  classifier->SetNumberOfClasses(numClasses);

  TFuzzyClassifier2D::CentroidArrayType centroidsArray;

  int argvIndex = 10;
  for (int i = 0; i < numClasses; i++)
    {
      centroidsArray.push_back(atof(argv[argvIndex]));
      ++argvIndex;
    }

  classifier->SetCentroids(centroidsArray);
  itk::SimpleFilterWatcher watcher(classifier, "MSKFCM classifier");

  typedef TFuzzyClassifier2D::CentroidType TCentroid;
  typedef itk::Statistics::RBFKernelInducedDistanceMetric<TCentroid>
      RBFKernelType;
  RBFKernelType::Pointer kernelDistancePtr = RBFKernelType::New();
  kernelDistancePtr->SetA(2.0);
  kernelDistancePtr->SetB(1.0);
  kernelDistancePtr->SetSigma(atoi(argv[argvIndex]));
  classifier->SetKernelDistanceMetric(static_cast<
      TClassifierMSKFCM::KernelDistanceMetricPointer >( kernelDistancePtr));

  typedef itk::FlatStructuringElement<dim> StructuringElement2DType;
  StructuringElement2DType::RadiusType elementRadius;
  for (int i = 0; i < dim; i++)
    {
      ++argvIndex;
      elementRadius[i] = atoi(argv[argvIndex]);
    }
  StructuringElement2DType structuringElement = StructuringElement2DType::Ball(
      elementRadius);
  classifier->SetStructuringElement(structuringElement);

  if ( (argc-1 > argvIndex ) && (strcmp(argv[argvIndex + 1], "-f") == 0) )
    {
      classifier->SetIgnoreBackgroundPixels(true);
      classifier->SetBackgroundPixel(atof(argv[argvIndex + 2]));
    }

  classifier->SetInput(reader->GetOutput());

  typedef itk::FuzzyClassifierImageFilter<TFuzzyClassifier::OutputImageType>
      TLabelClassifier2D;
  TLabelClassifier2D::Pointer labelClass = TLabelClassifier2D::New();
  labelClass->SetInput(classifier->GetOutput());

  typedef itk::ImageSeriesWriter<OType,OType2D> DicomWriterType;
  DicomWriterType::Pointer writer = DicomWriterType::New();
  writer->SetInput(labelClass->GetOutput());
  writer->SetImageIO(dicomIO);
  namesGenerator->SetOutputDirectory(argv[2]);
  writer->SetFileNames(namesGenerator->GetOutputFileNames());
  writer->SetMetaDataDictionaryArray(reader->GetMetaDataDictionaryArray());

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
