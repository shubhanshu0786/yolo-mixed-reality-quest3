/*
 * DetectionVisualizer.cs
 *
 * Example: subscribes to YOLODetectionClient.OnDetectionsReceived and
 * logs or acts on the results. Replace the body of HandleDetections with
 * your own VR-specific logic (spawn world-space labels, highlight objects, etc.)
 *
 * Attach to any GameObject alongside or separate from YOLODetectionClient.
 */

using System.Collections.Generic;
using UnityEngine;

public class DetectionVisualizer : MonoBehaviour
{
    void OnEnable()  => YOLODetectionClient.OnDetectionsReceived += HandleDetections;
    void OnDisable() => YOLODetectionClient.OnDetectionsReceived -= HandleDetections;

    void HandleDetections(DetectionResponse response)
    {
        foreach (var det in response.detections)
        {
            // `det.bbox` is in pixel space of the captured frame.
            // To map to screen/world space, normalize by frame_size, then
            // use Camera.main.ViewportToWorldPoint() or similar.

            // Example: log each unique class detected this frame
            Debug.Log($"  → {det.class_name} | conf={det.confidence:P1} | "
                      + $"bbox=({det.bbox.x1:F0},{det.bbox.y1:F0},{det.bbox.x2:F0},{det.bbox.y2:F0})");
        }
    }
}
