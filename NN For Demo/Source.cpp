// TrainNetwork.cpp : Defines the entry point for the console application.
 
#include "opencv2/opencv.hpp"    // opencv general include file
#include "opencv2/ml/ml.hpp"          // opencv machine learning include file
#include <stdio.h>
#include <fstream>
#include "dirent.h"
using namespace std;
/******************************************************************************/
 
//#define TRAINING_SAMPLES 3050       //Number of samples in training dataset
#define ATTRIBUTES 28*28  //Number of pixels per sample.16X16
//#define TEST_SAMPLES 1170       //Number of samples in test dataset
#define CLASSES 8                  //Number of distinct labels.
#define IMGDIM 28 


void read_image_set(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{
	DIR *dir;
    struct dirent *ent;
	vector<string> out;
	/* open directory stream */
	dir = opendir (filename);
    if (dir != NULL) {

      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL) {
        //printf ("%s\n", ent->d_name);
		out.push_back(ent->d_name);
      }
      closedir (dir);

    }

	 //read each row of the csv file
	int count =0;
	


	for(int row = 2; row < out.size(); row++)
   {
	   if(out[row].find("jpg")!=std::string::npos)
	   {
		   cout<< filename+out[row] <<endl;
		   
		   cv::Mat Img = cv::imread(filename+out[row]);
		   cout<< row <<endl;
		   //cv::imshow("asd",Img);
		   resize(Img,Img, cv::Size(IMGDIM,IMGDIM));
		   
		   cvtColor(Img,Img,CV_BGR2GRAY);
		   //imshow("out",Img);
		   //cv::waitKey(1000);
		   for (int i = 0; i < Img.rows; i++)
		   {
			   for (int j = 0; j < Img.cols; j++)
			   {
				   data.at<float>(count,i*IMGDIM+j) = Img.at<uchar>(i,j);
			   }
		   }
		   string path = filename + out[row].substr(0,out[row].size()-3)+"txt";
		   FILE* inputfile = fopen( path.c_str(), "r" );
		   int val;
		   fscanf(inputfile, "%i,", &val);
		   classes.at<float>(count,val-1) = 1.0;
		   count++;

		   if(count>=total_samples)
		   {
			   break;
		   }
		   fclose(inputfile);
	   }
    }
}
void read_image(char *filename, cv::Mat &data)
{		   
	cv::Mat Img = cv::imread(filename);
	resize(Img,Img, cv::Size(IMGDIM,IMGDIM));
		   
	cvtColor(Img,Img,CV_BGR2GRAY);
	// imshow("out",Img);
	//  waitKey(1000);

	for (int i = 0; i < Img.rows; i++)
	{
		for (int j = 0; j < Img.cols; j++)
		{
			data.at<float>(0,i*IMGDIM+j) = Img.at<uchar>(i,j);
		}
	}
}
void train(char *dataSetFileName, char *outputFileName, int n)
{
	 //matrix to hold the training sample
    cv::Mat training_set(n,ATTRIBUTES,CV_32F);
    //matrix to hold the labels of each taining sample
    cv::Mat training_set_classifications(n, CLASSES, CV_32F);
    //matric to hold the test samples
    cv::Mat test_set(n,ATTRIBUTES,CV_32F);
    //matrix to hold the test labels.
    cv::Mat test_set_classifications(n,CLASSES,CV_32F);
 
    //
    cv::Mat classificationResult(1, CLASSES, CV_32F);
    //load the training and test data sets.
	read_image_set(dataSetFileName, training_set, training_set_classifications, n);
	read_image_set(dataSetFileName, test_set, test_set_classifications, n);
 
        // define the structure for the neural network (MLP)
        // The neural network has 3 layers.
        // - one input node per attribute in a sample so 256 input nodes
        // - 16 hidden nodes
        // - 10 output node, one for each class.
 
        cv::Mat layers(4,1,CV_32S);
        layers.at<int>(0,0) = ATTRIBUTES;//input layer
        layers.at<int>(1,0)=100;//hidden layer
		layers.at<int>(2,0)=100;//hidden layer
        layers.at<int>(3,0) =CLASSES;//output layer
 
        //create the neural network.
        //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
        CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);
 
        CvANN_MLP_TrainParams params(                                  
 
                                        // terminate the training after either 1000
                                        // iterations or a very small change in the
                                        // network wieghts below the specified value
                                        cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
                                        // use backpropogation for training
                                        CvANN_MLP_TrainParams::BACKPROP,
                                        // co-efficents for backpropogation training
                                        // recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
                                        0.1,
                                        0.1);
 
        // train the neural network (using training data)
 
        printf( "\nUsing training dataset\n");
        int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
        printf( "Training iterations: %i\n\n", iterations);
 
        // Save the model generated into an xml file.
        CvFileStorage* storage = cvOpenFileStorage( outputFileName, 0, CV_STORAGE_WRITE );
        nnetwork.write(storage,"DigitOCR");
        cvReleaseFileStorage(&storage);
 
        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;
 
        //classification matrix gives the count of classes to which the samples were classified.
        int classification_matrix[CLASSES][CLASSES]={{}};
 
        // for each sample in the test set.
        for (int tsample = 0; tsample < n; tsample++) {
 
            // extract the sample
 
            test_sample = test_set.row(tsample);
 
            //try to predict its class
 
            nnetwork.predict(test_sample, classificationResult);
            /*The classification result matrix holds weightage  of each class.
            we take the class with the highest weightage as the resultant class */
 
            // find the class with maximum weightage.
            int maxIndex = 0;
            float value=0.0f;
            float maxValue=classificationResult.at<float>(0,0);
            for(int index=1;index<CLASSES;index++)
            {   value = classificationResult.at<float>(0,index);
                if(value>maxValue)
                {   maxValue = value;
                    maxIndex=index;
 
                }
            }
 
            printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
 
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            if (test_set_classifications.at<float>(tsample, maxIndex)!=1.0f)
            {
                // if they differ more than floating point error => wrong class
 
                wrong_class++;
 
                //find the actual label 'class_index'
                for(int class_index=0;class_index<CLASSES;class_index++)
                {
                    if(test_set_classifications.at<float>(tsample, class_index)==1.0f)
                    {
 
                        classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
                        break;
                    }
                }
 
            } else {
 
                // otherwise correct
 
                correct_class++;
                classification_matrix[maxIndex][maxIndex]++;
            }
        }
 
        printf( "\nResults on the testing dataset\n"
        "\tCorrect classification: %d (%g%%)\n"
        "\tWrong classifications: %d (%g%%)\n", 
        correct_class, (double) correct_class*100/n,
        wrong_class, (double) wrong_class*100/n);
        cout<<"   ";
        for (int i = 0; i < CLASSES; i++)
        {
            cout<< i<<"\t";
        }
        cout<<"\n";
        for(int row=0;row<CLASSES;row++)
        {cout<<row<<"  ";
            for(int col=0;col<CLASSES;col++)
            {
                cout<<classification_matrix[row][col]<<"\t";
            }
            cout<<"\n";
        } 
        return ;
}
int test(char *paramFileName, char *imagename)
{
	//read the model from the XML file and create the neural network.
    CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( paramFileName, 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);
 
    //your code here
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel
    // ... data for the digit to be recognized
    // ...
 
	cv::Mat data(1,ATTRIBUTES,CV_32F);

	read_image(imagename,data);

    int maxIndex = 0;
    cv::Mat classOut(1,CLASSES,CV_32F);
    //prediction
    nnetwork.predict(data, classOut);
    float value;
    float maxValue=classOut.at<float>(0,0);
    for(int index=1;index<CLASSES;index++)
    {   value = classOut.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex=index;
            }
    }
	return maxIndex;
    //maxIndex is the predicted class.
}

int test_image(char *paramFileName, cv::Mat img)
{
	//read the model from the XML file and create the neural network.
    CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( paramFileName, 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);
 
    //your code here
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel
    // ... data for the digit to be recognized
    // ...
 
	cv::Mat data(1,ATTRIBUTES,CV_32F);

	cv::resize(img,img, cv::Size(IMGDIM,IMGDIM));
		   
	cv::cvtColor(img,img,CV_BGR2GRAY);
	// imshow("out",img);
	//  waitKey(1000);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			data.at<float>(0,i*IMGDIM+j) = img.at<uchar>(i,j);
		}
	}

    int maxIndex = 0;
    cv::Mat classOut(1,CLASSES,CV_32F);
    //prediction
    nnetwork.predict(data, classOut);
    float value;
    float maxValue=classOut.at<float>(0,0);
    for(int index=1;index<CLASSES;index++)
    {   value = classOut.at<float>(0,index);
            if(value>maxValue)
            {   maxValue = value;
                maxIndex=index;
            }
    }
	return maxIndex;
}

void test_vedio(char *paramFileName)
{
	cv::VideoCapture cap(0);
	cv::Mat img;
	while (1)
	{
		cap >> img;

		if(!img.empty())
		{
			int ans  = test_image(paramFileName,img);

			cv::putText(img, std::to_string(ans), cv::Point(50,50), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 2, cv::Scalar(255,0,0));
			cv::imshow("out",img);
			cv::waitKey(30);
		}
	}
}
int main( int argc, char** argv )
{

	train("F:/Documents/Projects/Final Year Project/Shadows/data_set/", "C:/Users/Chameera/Desktop/param.xml", 5348/2);
	cv::waitKey(10000);
	int a = test("C:/Users/Chameera/Desktop/param.xml", "F:/Documents/Projects/Final Year Project/Shadows/Aduwa/6.jpg");

	cout<<"anser should be 4 :D"<<endl;
	cout<< a<< endl;
	getchar();

	//test_vedio("C:/Users/Chameera/Desktop/param_all_16.xml");
}