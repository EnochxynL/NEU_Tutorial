
// NORM_LANDMARKS is either the full set of landmarks for the hand, or
// a subset of the hand landmarks (indices 0, 1, 2, 3, 5, 6, 9, 10, 13, 14,
// 17 and 18). The latter is the legacy behavior, please just pass in
// the full set of hand landmarks.
//
// TODO: update clients to just pass all the landmarks in.
constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
// Indices within the partial landmarks.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr int kNumLandmarks = 21;
constexpr float kTargetAngle = M_PI * 0.5f;

inline float NormalizeRadians(float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

float ComputeRotation(const NormalizedLandmarkList &landmarks,
                      const std::pair<int, int> &image_size)
{
    const float x0 = landmarks.landmark(kWristJoint).x() * image_size.first;
    const float y0 = landmarks.landmark(kWristJoint).y() * image_size.second;

    float x1 = (landmarks.landmark(kIndexFingerPIPJoint).x() +
                landmarks.landmark(kRingFingerPIPJoint).x()) /
               2.f;
    float y1 = (landmarks.landmark(kIndexFingerPIPJoint).y() +
                landmarks.landmark(kRingFingerPIPJoint).y()) /
               2.f;
    x1 = (x1 + landmarks.landmark(kMiddleFingerPIPJoint).x()) / 2.f *
         image_size.first;
    y1 = (y1 + landmarks.landmark(kMiddleFingerPIPJoint).y()) / 2.f *
         image_size.second;

    const float rotation =
        NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
    return rotation;
}

absl::Status NormalizedLandmarkListToRect(
    const NormalizedLandmarkList &landmarks,
    const std::pair<int, int> &image_size, NormalizedRect *rect)
{
    const float rotation = ComputeRotation(landmarks, image_size);
    const float reverse_angle = NormalizeRadians(-rotation);

    // Find boundaries of landmarks.
    float max_x = std::numeric_limits<float>::min();
    float max_y = std::numeric_limits<float>::min();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < landmarks.landmark_size(); ++i)
    {
        max_x = std::max(max_x, landmarks.landmark(i).x());
        max_y = std::max(max_y, landmarks.landmark(i).y());
        min_x = std::min(min_x, landmarks.landmark(i).x());
        min_y = std::min(min_y, landmarks.landmark(i).y());
    }
    const float axis_aligned_center_x = (max_x + min_x) / 2.f;
    const float axis_aligned_center_y = (max_y + min_y) / 2.f;

    // Find boundaries of rotated landmarks.
    max_x = std::numeric_limits<float>::min();
    max_y = std::numeric_limits<float>::min();
    min_x = std::numeric_limits<float>::max();
    min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < landmarks.landmark_size(); ++i)
    {
        const float original_x =
            (landmarks.landmark(i).x() - axis_aligned_center_x) * image_size.first;
        const float original_y =
            (landmarks.landmark(i).y() - axis_aligned_center_y) * image_size.second;

        const float projected_x = original_x * std::cos(reverse_angle) -
                                  original_y * std::sin(reverse_angle);
        const float projected_y = original_x * std::sin(reverse_angle) +
                                  original_y * std::cos(reverse_angle);

        max_x = std::max(max_x, projected_x);
        max_y = std::max(max_y, projected_y);
        min_x = std::min(min_x, projected_x);
        min_y = std::min(min_y, projected_y);
    }
    const float projected_center_x = (max_x + min_x) / 2.f;
    const float projected_center_y = (max_y + min_y) / 2.f;

    const float center_x = projected_center_x * std::cos(rotation) -
                           projected_center_y * std::sin(rotation) +
                           image_size.first * axis_aligned_center_x;
    const float center_y = projected_center_x * std::sin(rotation) +
                           projected_center_y * std::cos(rotation) +
                           image_size.second * axis_aligned_center_y;
    const float width = (max_x - min_x) / image_size.first;
    const float height = (max_y - min_y) / image_size.second;

    rect->set_x_center(center_x / image_size.first);
    rect->set_y_center(center_y / image_size.second);
    rect->set_width(width);
    rect->set_height(height);
    rect->set_rotation(rotation);

    return absl::OkStatus();
}