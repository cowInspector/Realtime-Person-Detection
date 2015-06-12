/*
Build on the previous assignments where you web_cameratured a video-clip in real-time using a video camera. 
For this final project, please do the following:
1.	Work on real-time video (i.e., not web_cameratured video)
2.	Video should have few persons (say 2 to 4) moving around and one or two of them should be wearing 
    UTD logo T-shirts.
3.	UTD T-shirts do have different types of UTD logos. You can choose to track any one (or more) of these 
    logos. The choice of logos is yours. But during the demo you should have at least 1 person wearing that 
	logo T-shirt (and that can be you yourself).
4.	Of the few persons seen on the video, track only the person wearing UTD T-shirt, i.e., put a bounding box 
    on the entire person wearing the T-shirt (not just the T-shirt alone).
5.	Bounding box on the entire person wearing T-shirt is a requirement.
6.	At the end (which can be decided by just pressing a key), display a graphic representation of how the UTD 
    T-shirt person was moving, i.e., the path of the person’s movement. This can be done by displaying a rectangular
	box (for the web_cameratured video space) and drawing lines that would connect the points of the center of the bounding
	box in different video frames.
7.	Only one such graphic representation of the UTD T-shirt person’s movement path is needed – this path will show the
    cumulative movement over the entire video clip web_cameratured.

Author: Yogeshwara Krishnan
Last Modified: 05/12/2014

DECLARATION:
0. OpenCV is released under BSD license and hence it is free for academic and commercial use.
1. Main code snippet that extracts the Bag of Words keypoints from the images
   is from the OpenCV documentation and tutorials.
2. The HoG descriptor that detects humans is also from the OpenCV documentation
   and tutorials.
3. The above mentioned code snippets are available in public domain and are free to reuse.
4. The rest of the code which involves pre and post-processing of the training data, prediction, path tracing was solely written
   by the author mentioned above. 
*/

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

/* Function declarations. */
NormalBayesClassifier train_data();
void detect_humans(NormalBayesClassifier&);
void get_training_vocabulary();
void get_BOW_Descriptor(Mat&, Mat&);
void get_descriptor_from_rect(Mat&, Mat&, Mat&);

// Declare a Flann based descriptor matcher.
Ptr<DescriptorMatcher> flann_matcher = DescriptorMatcher::create("FlannBased");
// Declare a SURF decriptor extractor.
Ptr<DescriptorExtractor> surf_extractor = DescriptorExtractor::create("SURF");
// Declare a SURF feature detector.
Ptr<FeatureDetector> surf_detector = FeatureDetector::create("SURF");
// Declare a global vector to store the keypoints in an image
vector<KeyPoint> keypoints;
// Set the dictionary size. These many features will be used for training.
int dictionary_size = 400;
// The trainer needs a termination criteria during iteration.
TermCriteria termination_criteria(CV_TERMCRIT_ITER, 20, 0.002);
// Maximum no. of retries for the training algorithm
int no_of_retries = 2;
// This flag tells the trainer to use KMEANS++ cluster algorithm to detect clusters.
int flags = KMEANS_PP_CENTERS;

// Declare a Bag of Words trainer. 
// Dictionary size, termination criteria, no. of retries and flags are passed for initialization.
BOWKMeansTrainer bow_trainer(dictionary_size, termination_criteria, no_of_retries, flags);
// Declare a Bag of Words descriptor extractor. Tell it to use SURF extractor along with Flann matcher.
BOWImgDescriptorExtractor bow_desc_extractor(surf_extractor, flann_matcher);

// The program starts here.
int main(int argc, char **argv) {
	// Initialize the non-free modules such as SURF.
	cv::initModule_nonfree();
	// Train the images and use a Normal Bayes classifier.
	NormalBayesClassifier bayes_c = train_data();
	// Detect people and pass the bayes classifier to check if the person is wearing the UTD t-shirt.
	detect_humans(bayes_c);
	return 0;
}

// Function that trains the features extracted from the test images.
NormalBayesClassifier train_data() {
	cout << "DEBUG: Extracting the training vocabulary..." << endl;
	// Get the training vocabulary.
	get_training_vocabulary();
	// Use KMEANS++ clustering and cluster the keypoints.
	cout << "DEBUG: Using KMEANS++ clustering..." << endl;
	Mat dictionary = bow_trainer.cluster();
	// Set the vocabulary after clustering.
	bow_desc_extractor.setVocabulary(dictionary);
	// Create a Mat for the training data. SURF uses 32-bit float descriptors.
	Mat training_data(0, dictionary_size, CV_32FC1);
	// Create a Mat for the labels given to the images during classification.
	Mat labels(0, 1, CV_32FC1);
	cout << "DEBUG: Getting Bag of Words descriptor...\n";
	// Get Bag of Words decriptors for the training data and labels.
	get_BOW_Descriptor(training_data, labels);
	// Create a Normal Bayes classifier.
	NormalBayesClassifier bayes_classifier;
	cout << "DEBUG: Training the classifier...\n";
	// Train the classifier.
	bayes_classifier.train(training_data, labels);
	return bayes_classifier;
}

// Function that detects persons in the frame.
void detect_humans(NormalBayesClassifier& bayes_classifier) {
	// Access the web cam to capture the frame.
	VideoCapture web_camera(0);
	// Check if web cam is accessible.
	if (!web_camera.isOpened())
        exit(1);
	// Store the feed from the web cam into a Mat object.
	Mat frame, path_mat;
	// Declare a HOG descriptor for person detection.
    HOGDescriptor hog_descriptor;
	// Tell the HoG descriptor to use SVM and default people detector.
	hog_descriptor.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	// Get the video feed's width and height.
	int width = web_camera.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = web_camera.get(CV_CAP_PROP_FRAME_HEIGHT);
	// Create windows for display.
	namedWindow("video capture", CV_WINDOW_AUTOSIZE);
	namedWindow("detections", CV_WINDOW_AUTOSIZE);
	// Declare a vector of Point to store the path of the person's movement.
	vector<Point> path_coordinates;
	// Loop until stopped.
	while (true)
    {
		Mat results;
		// Create a Mat for evaluating the frame from the video feed.
		Mat evaluation_data(0, dictionary_size, CV_32FC1);
		// Create a Mat to check for the labels.
		Mat ground_truth(0, 1, CV_32FC1);
		// Read the web cam feed to a Mat.
        web_camera >> frame;
		path_mat = frame;
		// Check if the feed has data.
        if (!frame.data)
            continue;
 
		//flip(frame, frame, 1);
		// Create vector to store the area enclosing the detected person. 
        vector<Rect> found, person_found;
		// Start detecting the persons present in the frame. The scale factor of 1.05 is the optimal choice.
		hog_descriptor.detectMultiScale(frame, person_found, 0, Size(4,4), Size(32,32), 1.05, 2.0);
		// For each person detected, check if he is wearing the UTD t-shirt or not.
        for (int i=0; i<person_found.size(); i++) {
			// Find the enclosing area.
			Rect human = person_found[i];
			// Scale down the enclosing window.
			human.x += cvRound(human.width*0.1);
			human.width = cvRound(human.width*0.8);
			human.y += cvRound(human.height*0.06);
			human.height = cvRound(human.height*0.9);
			// Make sure that the enclosing rectangle is within the frame's dimensions.
			if (human.x < 0) {
				human.width += human.x;
				human.x = 0;
			} else if (human.x + human.width > width) {
				human.width = width - human.x;
			}
			
			if (human.y < 0) {
				human.height += human.y;
				human.y = 0;
			} else if (human.y + human.height > height) {
				human.height = height - human.y;
			}

			/* Restrict the area to be detected. HoG detects hands, legs, and partial body parts.
			   We are interested in presence of full body.
			   Restricting the area reduces the overall computation time.*/
			if (human.area() > 25000) { 
				// Extract the set of pixels where a person has been detected.
				Mat frame_rect(frame, human);
				cv::addWeighted(frame_rect, 1.5, frame_rect, -0.5, 0, frame_rect);
				// Get the descriptors from this rectangular area.
				get_descriptor_from_rect(frame_rect, evaluation_data, ground_truth);
				// Use the Bayes classifier to predict if the person is wearing the UTD t-shirt or not.
				float class_result = bayes_classifier.predict(evaluation_data, &results);
				cout << "Class: " << class_result << endl;
				// If the person detected is wearing the t-shirt with UTD logo...
				if (class_result == 1) {
					//cout << "Area: " << human.area() << endl;
					// Draw the bounding rectangle around the person.
					rectangle(frame, human.tl(), human.br(), cv::Scalar(0,0,255), 2);
					//cout << r.x + (r.width/2) << " , " << r.y + (r.height / 2) << endl;
					// Get the centre of the rectangle and store it in a vector.
					Point p(human.x + (human.width/2), human.y + (human.height / 2));
					path_coordinates.push_back(p);
					imshow("detections", frame_rect);
				}
			}
		}
        
		// Trace the path when the escape key is pressed.
        if (waitKey(20) == 27) {
			cout << "DEBUG: Escape key pressed..." << endl;
			// Check the vector size. Trace the path only if the person has moved.
			if (path_coordinates.size() > 1) {
				Point start = path_coordinates.at(0); // Get the starting point.
				cout << start.x << "," << start.y << endl;
				// Loop through all the points and connect them through the lines.
				for (int i = 1; i < path_coordinates.size(); i++) {
					Point inter = path_coordinates.at(i);
					//line(frame, start, inter, cv::Scalar(255, 0, 0), 2);
					line(path_mat, start, inter, cv::Scalar(255, 0, 0), 2);
					start = inter;
				}
			}
			imshow("video capture", path_mat);
			//break;
		}
		imshow("video capture", frame);
    }
	//imshow("video capture", frame);
}

/*
* Extract the training vocabulary. Input is a text file that has labels and path to
* the image file.
*/
void get_training_vocabulary() {
	string line;
	int label;
	string file_name;
	ifstream input_file("train.txt"); // Open the text file.
	if (input_file.fail()) {
		cout << "Error opening the file.\n";
		exit(1);
	}
	else {
		// Process each line of the text file.
		while (getline(input_file, line)) {
			std::istringstream ss(line);
			ss >> label >> file_name;
			cout << "Processing file " << file_name << endl;
			// Read the image file.
			Mat frame = imread(file_name);
			if (!frame.empty()) { // If image isn't empty.
				surf_detector->detect(frame, keypoints); // Get the keypoints.
				if (keypoints.empty()) {
						cerr << "Warning: Could not find key points in image: "
										<< file_name << endl;
				} else {
						Mat features;
						// Extract the SURF features from the frame and add it to the trainer.
						surf_extractor->compute(frame, keypoints, features);
						bow_trainer.add(features);
				}
			} else {
				cerr << "Warning: Could not read image: " << file_name << endl;
			}
		}
		input_file.close(); // Close the file.
	}
}

/*
* Function to extract BoW descriptors.
*/
void get_BOW_Descriptor(Mat& descriptors, Mat& labels) {
	string line;
	int class_label;
	string file_name;
	ifstream input_file("train.txt"); // Open the text file
	if (input_file.fail()) {
		cout << "Error opening the file.\n";
		exit(1);
	}
	else {
		// Process each line of the text file.
		while (getline(input_file, line)) {
			std::istringstream ss(line);
			ss >> class_label >> file_name;
			cout << "Processing file " << file_name << endl;
			Mat frame = imread(file_name); // Read the image.
			if (!frame.empty()) {
				// Extract the SURF detectors.
				surf_detector->detect(frame, keypoints);
				if (keypoints.empty()) {
					cerr << "Warning: Could not find key points in image: " << file_name << endl;
				} else {
					Mat bowDescriptor;
					// Compute the BoW extractors.
					bow_desc_extractor.compute(frame, keypoints, bowDescriptor);
					descriptors.push_back(bowDescriptor);
					labels.push_back(class_label);
				}
			} else {
				cerr << "Warning: Could not read image: " << file_name << endl;
			}
		}
	}
}

/*
* Function to extract BoW descriptor from the area enclosing detected person.
*/
void get_descriptor_from_rect(Mat &rect, Mat& descriptors, Mat& labels) {
	if (!rect.empty()) {
		// Get the SURF features from the rect.
		surf_detector->detect(rect, keypoints);
		if (keypoints.empty()) {
			cerr << "Warning: Could not find key points in image." << endl;
		} else {
			Mat bowDescriptor;
			// Extract the BoW descriptors and add them to Mat along with the labels.
			bow_desc_extractor.compute(rect, keypoints, bowDescriptor);
			descriptors.push_back(bowDescriptor);
			labels.push_back(90);
		}
	} else {
		cerr << "Warning: Could not read frame." << endl;
	}
}