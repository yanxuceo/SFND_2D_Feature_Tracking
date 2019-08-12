## Camera Based 2D Feature Tracking

### Part I: Solution Description
#### MP.1 Data Buffer Optimization
Implements a ring buffer where new elements are added to tail and older are removed from head. And in this case, set the capacity of this buffer as 2, which represents two adjacent frame reading from sequences of traffic images. Part of a template class definition can be seen below.

```
template<typename T, unsigned int kMaxCapacity>
class RingBuffer
{
  public:
    typedef unsigned int SizeType;
    typedef T ValueType;
    typedef ValueType& Reference;
    typedef const ValueType& ConstReference;
    
    static const SizeType kCapacityLimit = kMaxCapacity;

    explicit RingBuffer(SizeType capacity = kMaxCapacity)
        :head_(0U), tail_(0U), size_(0U), capacity_(capacity)
    {
        SetMaxSize(capacity);
    }

    ~RingBuffer() {}

    // ignore concrete member functions here
}
```

#### MP.2 Keypoint Detection
Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.


1. Traditional Harris detector for keypoints detection is given by this function. 
```
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;      // a blockSize x blockSize neighborhood for every pixel
    int apertureSize = 3;   // for sobel operator
    int minResponse = 100;  // minimum value for a corner in the 8-bit scaled response matrix
    double k = 0.04;        // Harris parameter

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // look for prominent corners and keypoints
    double maxOverlap = 0.0;
    for(size_t i = 0; i < dst_norm.rows; i++)
    {
        for(size_t j = 0; j < dst_norm.cols; j++)
        {
            int response = (int)dst_norm.at<float>(i,j);
            if(response > minResponse)
            {
                // only store points above a threshold
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(j, i);
                newKeypoint.size = 2*apertureSize;
                newKeypoint.response = response;

                // ignore codes below for perform non-maximal suppression
            }
        }
    }
}
```

2. The other modern detector including FAST, BRISK, ORB, AKAZE, and SIFT are given in this function below, in which a parameter called **_detectorType_** makes them selectable.
```
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;     
        int bNMS = true;      
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
       
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("ORB") == 0)
    {   
        detector = cv::ORB::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
        detector->detect(img, keypoints);
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Modern Detector Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);

        cv::waitKey(0);
    }
}
```

#### MP.3 Keypoint Removal
To remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. A lambda expression is used to loop through all detected keypoints, and  _std::vector::erase_, _std::remove_if_, _cv::Rect::contains_ are incorporated to finish this job.

```
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
if (bFocusOnVehicle)
{
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [&vehicleRect](const cv::KeyPoint &r){return !(vehicleRect.contains(r.pt));}),
                keypoints.end());
}
```

#### MP.4 Keypoint Descriptors
Implements descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly. Similar with above keypoint detection, a miscellaneous function including BRIEF, ORB, FREAK, AKAZE and SIFT descriptors are given in this function, in which a parameter called **_detectorType_** makes them selectable.

```
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {
        extractor = cv::BRISK::create();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }

    // perform feature description
    extractor->compute(img, keypoints, descriptors);
}

```

#### MP.5 Descriptor Matching && MP.6 Descriptor Distance Ratio
Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function; Use the KNN matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

All these three tasks are realized in this function, k = 2; distance ratio = 0.8;
```
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { 
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { 
        // Finds the best match for each descriptor in desc1
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k = 2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
      
        // Implement k-nearest-neighbor matching and filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        std::cout << "# keypoints removed = " << knn_matches.size() - matches.size() << std::endl;
    }
}
```


### Part II: Performance Evaluation

#### MP.7 Keypoints Counting
To count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.


| Detector |Img 0|Img 1|Img 2|Img 3|Img 4|Img 5|Img 6|Img 7|Img 8|Img 9|
| ---      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harris   | 17  | 14  | 18  | 21  | 26  | 43  | 18  | 31  | 26  | 34  |
|Shi-Tomasi| 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 |
| FAST     | 149 | 152 | 150 | 155 | 149 | 149 | 156 | 150 | 138 | 143 |
| BRISK    | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 266 | 254 |
| ORB      | 92  | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 |
| AKAZE    | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 |
| SIFT     | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 |


Harris, Shi-Tomasi and FAST has similar, relatively small neighborhood size; and they are distributed spacially, no overlap with each other.



BRISK, however, has obvious large neighborhood size; and they look like cluttered and overlapped with each other.

ORB

AKAZE

SIFT

 
#### MP.8 Matching Statistics
To count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, use the BF approach with the descriptor distance ratio set to 0.8.

**_Note_**
L1 and L2 norms are preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK, BRIEF, FREAK and AKAZE. NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description).

| Combination(detect + descriptor)| # Matched Keypoints | Detection Time | Extraction Time | Matching Time |
| ---                             | ---                 | ---            | ---             | ---           |
| **_Group 1_**                   |                     |                |                 |               |
| Harris + SIFT                   |                     |                |                 |               |
| Harris + BRISK                  |                     |                |                 |               |
| Harris + ORB                    |                     |                |                 |               |
| Harris + FREAK                  |                     |                |                 |               | 
| Harris + AKAZE                  |                     |                |                 |               |
| Harris + BRIEF                  |                     |                |                 |               |
| **_Group 2_**                   |                     |                |                 |               | 
| Shi-Tomasi + SIFT               |                     |                |                 |               |
| Shi-Tomasi + BRISK              |                     |                |                 |               |
| Shi-Tomasi + ORB                |                     |                |                 |               |
| Shi-Tomasi + FREAK              |                     |                |                 |               | 
| Shi-Tomasi + AKAZE              |                     |                |                 |               |
| Shi-Tomasi + BRIEF              |                     |                |                 |               |
| **_Group 3_**                   |                     |                |                 |               | 
| FAST + SIFT                     |                     |                |                 |               |
| FAST + BRISK                    |                     |                |                 |               |
| FAST + ORB                      |                     |                |                 |               |
| FAST + FREAK                    |                     |                |                 |               | 
| FAST + AKAZE                    |                     |                |                 |               |
| FAST + BRIEF                    |                     |                |                 |               |
| **_Group 4_**                   |                     |                |                 |               | 
| BRISK + SIFT                    |                     |                |                 |               |
| BRISK + BRISK                   |                     |                |                 |               |
| BRISK + ORB                     |                     |                |                 |               |
| BRISK + FREAK                   |                     |                |                 |               | 
| BRISK + AKAZE                   |                     |                |                 |               |
| BRISK + BRIEF                   |                     |                |                 |               |
| **_Group 5_**                   |                     |                |                 |               | 
| ORB + SIFT                      |                     |                |                 |               |
| ORB + BRISK                     |                     |                |                 |               |
| ORB + ORB                       |                     |                |                 |               |
| ORB + FREAK                     |                     |                |                 |               | 
| ORB + AKAZE                     |                     |                |                 |               |
| ORB + BRIEF                     |                     |                |                 |               |
| **_Group 6_**                   |                     |                |                 |               | 
| AKAZE + SIFT                    |                     |                |                 |               |
| AKAZE + BRISK                   |                     |                |                 |               |
| AKAZE + ORB                     |                     |                |                 |               |
| AKAZE + FREAK                   |                     |                |                 |               | 
| AKAZE + AKAZE                   |                     |                |                 |               |
| AKAZE + BRIEF                   |                     |                |                 |               |
| **_Group 7_**                   |                     |                |                 |               | 
| SIFT + SIFT                     |                     |                |                 |               |
| SIFT + BRISK                    |                     |                |                 |               |
| SIFT + ORB                      |                     |                |                 |               |
| SIFT + FREAK                    |                     |                |                 |               | 
| SIFT + AKAZE                    |                     |                |                 |               |
| SIFT + BRIEF                    |                     |                |                 |               |



#### MP.9 Time Consumption
To log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this information you will then suggest the TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles. Finally, in a short text, please justify your recommendation based on your observations and on the data you collected.