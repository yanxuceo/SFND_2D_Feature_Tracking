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

    ///
    /// @brief constructs a Ring Buffer
    ///
    /// @param[in] capacity maximum capacity that the buffer can hold. Must not exceed kMaxCapacity
    ///
    explicit RingBuffer(SizeType capacity = kMaxCapacity)
        :head_(0U), tail_(0U), size_(0U), capacity_(capacity)
    {
        SetMaxSize(capacity);
    }

    /// @brief destructor
    ~RingBuffer()
    {
    }
    ...
    ...
}
```

#### MP.2 Keypoint Detection


#### MP.3 Keypoint Removal
To remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. Here, a lambda expression is used to loop through all 

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
Implements descriptors

#### MP.5 Descriptor Matching
FLANN

#### MP.6 Descriptor Distance Ratio
For the K-Nearest-Neighbor matching,


### Part II: Performance Evaluation

#### MP.7 Keypoints Counting
Here 
 
#### MP.8 Matching Statistics
The matched keypoints

#### MP.9 Time Consumption
Time it takes