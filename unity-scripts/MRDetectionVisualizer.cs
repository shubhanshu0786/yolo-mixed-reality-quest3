/*
 * MRDetectionVisualizer.cs
 *
 * Quest 3 Mixed Reality integration for YOLODetectionClient.
 * Subscribes to detection events and spawns world-space 3D labels
 * at each detected object's estimated position using viewport raycasting.
 *
 * SETUP:
 *   1. Create a Label Prefab (see instructions below).
 *   2. Attach this script to any GameObject in your scene.
 *   3. Set `vrCamera` to: OVRCameraRig → TrackingSpace → CenterEyeAnchor
 *   4. Set frameWidth/frameHeight to match YOLODetectionClient's captureWidth/captureHeight.
 *
 * LABEL PREFAB SETUP:
 *   - Hierarchy → Right-click → Create Empty → name it "DetectionLabel"
 *   - Add a child: 3D Object → Quad  (scale ~0.2, 0.1, 1)
 *   - Add another child: UI → Text - TextMeshPro (set to 3D, World Space)
 *   - OR simply add a TextMeshPro (3D) component directly on a child object
 *   - Save as a prefab in your Assets folder
 */

using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class MRDetectionVisualizer : MonoBehaviour
{
    [Header("Prefab")]
    [Tooltip("World-space label prefab with a TextMeshPro component in its hierarchy.")]
    public GameObject labelPrefab;

    [Header("Camera")]
    [Tooltip("Center eye camera. Drag OVRCameraRig/TrackingSpace/CenterEyeAnchor here.")]
    public Camera vrCamera;

    [Header("Frame Size — must match YOLODetectionClient")]
    public int frameWidth  = 640;
    public int frameHeight = 480;

    [Header("Label Placement")]
    [Tooltip("Meters in front of camera to place a label when no scene geometry is hit.")]
    [Range(0.3f, 5f)]
    public float defaultDepth = 1.5f;

    [Tooltip("Raycast against Physics colliders to snap labels onto real surfaces.\n"
           + "Requires Quest Scene Understanding mesh to have colliders enabled.")]
    public bool usePhysicsRaycast = false;

    [Tooltip("Which layers to hit when raycasting (typically your room mesh layer).")]
    public LayerMask raycastLayers = ~0;

    [Header("Filtering")]
    [Tooltip("Detections below this confidence are skipped.")]
    [Range(0f, 1f)]
    public float minConfidence = 0.5f;

    [Tooltip("If non-empty, only show labels for these class names (e.g. person, chair).")]
    public List<string> showOnlyClasses = new();

    // ── Private ──────────────────────────────────────────────────────────────

    readonly List<GameObject> _activeLabels = new();

    void OnEnable()  => YOLODetectionClient.OnDetectionsReceived += HandleDetections;
    void OnDisable() => YOLODetectionClient.OnDetectionsReceived -= HandleDetections;

    void Start()
    {
        if (vrCamera == null)
            vrCamera = Camera.main;
    }

    // Called on the main thread by YOLODetectionClient every processed frame
    void HandleDetections(DetectionResponse response)
    {
        ClearLabels();

        if (labelPrefab == null || vrCamera == null)
            return;

        foreach (var det in response.detections)
        {
            if (det.confidence < minConfidence) continue;
            if (showOnlyClasses.Count > 0 && !showOnlyClasses.Contains(det.class_name)) continue;

            PlaceLabel(det);
        }
    }

    void PlaceLabel(Detection det)
    {
        // bbox center → normalized viewport [0,1]  (Unity Y is bottom-up, image Y is top-down)
        float vpX = det.center.x / frameWidth;
        float vpY = 1f - (det.center.y / frameHeight);

        Ray ray = vrCamera.ViewportPointToRay(new Vector3(vpX, vpY, 0f));

        Vector3 worldPos;
        if (usePhysicsRaycast && Physics.Raycast(ray, out RaycastHit hit, 5f, raycastLayers))
            worldPos = hit.point + hit.normal * 0.05f;   // slight offset so it floats off surface
        else
            worldPos = ray.origin + ray.direction * defaultDepth;

        var label = Instantiate(labelPrefab, worldPos, FacingCamera(worldPos));
        SetLabelText(label, det);
        _activeLabels.Add(label);
    }

    // Returns a rotation that makes the object face the camera (billboard)
    Quaternion FacingCamera(Vector3 from)
    {
        Vector3 dir = from - vrCamera.transform.position;
        return dir.sqrMagnitude > 0.001f
            ? Quaternion.LookRotation(dir)
            : Quaternion.identity;
    }

    void SetLabelText(GameObject label, Detection det)
    {
        var tmp = label.GetComponentInChildren<TextMeshPro>();
        if (tmp != null)
            tmp.text = $"<b>{det.class_name}</b>\n{det.confidence:P0}";

        // Also works for TextMeshProUGUI (world-space Canvas)
        var tmpUGUI = label.GetComponentInChildren<TextMeshProUGUI>();
        if (tmpUGUI != null)
            tmpUGUI.text = $"<b>{det.class_name}</b>\n{det.confidence:P0}";
    }

    void ClearLabels()
    {
        foreach (var l in _activeLabels)
            if (l != null) Destroy(l);
        _activeLabels.Clear();
    }

    void OnDestroy() => ClearLabels();
}
