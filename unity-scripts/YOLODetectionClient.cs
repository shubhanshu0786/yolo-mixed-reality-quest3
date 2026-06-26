/*
 * YOLODetectionClient.cs  —  v3 (OVRPlugin Passthrough Camera)
 *
 * THREE MODES — set CaptureMode in the Inspector:
 *
 *   Passthrough  — Uses Meta's PassthroughCameraAccess (MR Utility Kit) to grab
 *                  real passthrough camera frames from Quest 3 and send them to
 *                  the YOLO server. THIS IS THE CORRECT MODE for real-world detection.
 *                  Requires PassthroughCameraAccess component assigned below and
 *                  "horizonos.permission.HEADSET_CAMERA" in AndroidManifest.xml.
 *                  Set config.yaml → server.source = "unity"
 *
 *   Unity        — Captures from a Unity Camera (RenderTexture). Only sees virtual
 *                  objects, not the real world. Kept for testing.
 *                  Set config.yaml → server.source = "unity"
 *
 *   Webcam       — Server captures PC webcam, pushes detections to Quest.
 *                  Unity only listens; no frames are sent from here.
 *                  Set config.yaml → server.source = "webcam"
 *
 * SCENE SETUP (Passthrough mode):
 *   1. Add PassthroughCameraAccess component to YOLO Manager (or any GameObject).
 *   2. Set its Camera Position to "Left" (one of Quest 3's passthrough cameras).
 *   3. Assign it to the "Passthrough Camera" slot on this component.
 *   4. Set Capture Mode = Passthrough.
 *   5. Build as APK, run on Quest 3 — grant camera permission when prompted.
 */

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Android;
using NativeWebSocket;
using System.Text;
using Meta.XR;

// ── Data classes ─────────────────────────────────────────────────────────────

[Serializable] public class BBox { public float x1, y1, x2, y2; }
[Serializable] public class Vec2 { public float x, y; }
[Serializable] public class SizeInfo { public float width, height; }

[Serializable]
public class Detection
{
    public int      class_id;
    public string   class_name;
    public float    confidence;
    public BBox     bbox;
    public Vec2     center;
    public SizeInfo size;
}

[Serializable]
public class DetectionResponse
{
    public int             frame_id;
    public List<Detection> detections;
    public float           inference_time_ms;
}

// ── Main component ────────────────────────────────────────────────────────────

public class YOLODetectionClient : MonoBehaviour
{
    public enum CaptureMode { Passthrough, Unity, Webcam }

    [Header("Server")]
    public string serverUrl = "ws://192.168.31.143:8765";

    [Header("Mode")]
    [Tooltip("Passthrough: real-world frames via Quest 3 passthrough camera (recommended).\n"
           + "Unity: virtual camera RenderTexture (testing only).\n"
           + "Webcam: server pushes detections from PC webcam, Unity only listens.")]
    public CaptureMode captureMode = CaptureMode.Passthrough;

    [Header("Passthrough Camera (Quest 3)")]
    [Tooltip("Add PassthroughCameraAccess component to this GameObject and assign it here.")]
    public PassthroughCameraAccess passthroughCamera;

    [Header("Capture Settings")]
    [Tooltip("Frame width sent to YOLO server. Must match MRDetectionVisualizer.frameWidth.")]
    public int captureWidth  = 640;
    [Tooltip("Frame height sent to YOLO server. Must match MRDetectionVisualizer.frameHeight.")]
    public int captureHeight = 480;
    [Range(0.05f, 2f)] public float captureIntervalSeconds = 0.15f;
    [Range(1, 100)]    public int   jpegQuality = 75;

    [Header("Unity Mode Camera (fallback)")]
    [Tooltip("Only used when Capture Mode = Unity.")]
    public Camera captureCamera;

    [Header("Debug")]
    public bool logDetections = true;

    // ── Events ────────────────────────────────────────────────────────────────

    public static event Action<DetectionResponse> OnDetectionsReceived;

    // ── Private ───────────────────────────────────────────────────────────────

    const string HeadsetCameraPermission = "horizonos.permission.HEADSET_CAMERA";

    WebSocket     _socket;
    RenderTexture _renderTexture;
    Texture2D     _readbackTexture;

    bool  _waitingForResponse;
    float _nextCaptureTime;
    int   _sentFrameId;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void Awake()
    {
        // Request camera permission from Awake (sync, main thread) — reliable on Horizon OS.
        if (captureMode == CaptureMode.Passthrough &&
            !Permission.HasUserAuthorizedPermission(HeadsetCameraPermission))
        {
            Debug.Log("[YOLO] Requesting headset camera permission...");
            Permission.RequestUserPermission(HeadsetCameraPermission);
        }
    }

    async void Start()
    {
        if (captureMode == CaptureMode.Unity)
        {
            if (captureCamera == null) captureCamera = Camera.main;
            _renderTexture   = new RenderTexture(captureWidth, captureHeight, 24, RenderTextureFormat.ARGB32);
            _readbackTexture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        }

        _socket = new WebSocket(serverUrl);
        _socket.OnOpen    += () => Debug.Log("[YOLO] Connected to server.");
        _socket.OnError   += e  => Debug.LogWarning($"[YOLO] WebSocket error: {e}");
        _socket.OnClose   += c  => Debug.Log($"[YOLO] Connection closed: {c}");
        _socket.OnMessage += OnMessage;

        await _socket.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _socket?.DispatchMessageQueue();
#endif
        if (captureMode == CaptureMode.Webcam) return;
        if (_socket == null || _socket.State != WebSocketState.Open) return;
        if (_waitingForResponse) return;
        if (Time.time < _nextCaptureTime) return;

        _nextCaptureTime = Time.time + captureIntervalSeconds;

        if (captureMode == CaptureMode.Passthrough)
            CaptureFromPassthrough();
        else
            CaptureFromCamera();
    }

    async void OnApplicationQuit()
    {
        if (_socket != null) await _socket.Close();
    }

    // ── Passthrough capture (real world) ──────────────────────────────────────

    int _debugLogCounter;

    void CaptureFromPassthrough()
    {
        if (passthroughCamera == null)
        {
            Debug.LogError("[YOLO] PassthroughCamera not assigned in Inspector!");
            return;
        }
        if (!passthroughCamera.IsPlaying)
        {
            // Log every 60 frames (~4 sec) so logcat shows the block reason
            if (_debugLogCounter++ % 60 == 0)
            {
                bool hasPerm = Permission.HasUserAuthorizedPermission(HeadsetCameraPermission);
                Debug.LogWarning($"[YOLO] Camera not playing. Permission granted: {hasPerm}. Waiting...");
            }
            return;
        }

        Texture tex = passthroughCamera.GetTexture();
        if (tex == null) return;

        // Downscale the passthrough texture (1280x960) to the smaller send resolution.
        if (_renderTexture == null || _renderTexture.width != captureWidth || _renderTexture.height != captureHeight)
        {
            if (_renderTexture != null) _renderTexture.Release();
            _renderTexture = new RenderTexture(captureWidth, captureHeight, 0, RenderTextureFormat.ARGB32);
        }
        Graphics.Blit(tex, _renderTexture);

        if (_readbackTexture == null || _readbackTexture.width != captureWidth || _readbackTexture.height != captureHeight)
        {
            if (_readbackTexture != null) Destroy(_readbackTexture);
            _readbackTexture = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
        }

        var prevActive = RenderTexture.active;
        RenderTexture.active = _renderTexture;
        _readbackTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        _readbackTexture.Apply();
        RenderTexture.active = prevActive;

        byte[] jpegBytes = _readbackTexture.EncodeToJPG(jpegQuality);
        _waitingForResponse = true;
        _sentFrameId++;
        _ = _socket.Send(jpegBytes);
    }

    // ── Unity camera capture (virtual only — for testing) ────────────────────

    void CaptureFromCamera()
    {
        var prev = captureCamera.targetTexture;
        captureCamera.targetTexture = _renderTexture;
        captureCamera.Render();
        captureCamera.targetTexture = prev;

        var prevActive = RenderTexture.active;
        RenderTexture.active = _renderTexture;
        _readbackTexture.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
        _readbackTexture.Apply();
        RenderTexture.active = prevActive;

        byte[] jpegBytes = _readbackTexture.EncodeToJPG(jpegQuality);
        _waitingForResponse = true;
        _sentFrameId++;
        _ = _socket.Send(jpegBytes);
    }

    // ── Response handling ─────────────────────────────────────────────────────

    void OnMessage(byte[] rawBytes)
    {
        _waitingForResponse = false;

        string json = Encoding.UTF8.GetString(rawBytes);
        if (json.Contains("\"error\""))
        {
            Debug.LogWarning($"[YOLO] Server error: {json}");
            return;
        }

        DetectionResponse response;
        try { response = JsonUtility.FromJson<DetectionResponse>(json); }
        catch (Exception e) { Debug.LogError($"[YOLO] JSON parse error: {e.Message}"); return; }

        if (logDetections && response.detections.Count > 0)
        {
            var sb = new StringBuilder($"[YOLO] frame {response.frame_id} | {response.inference_time_ms:F1}ms | ");
            foreach (var d in response.detections)
                sb.Append($"{d.class_name}({d.confidence:P0}) ");
            Debug.Log(sb.ToString());
        }

        OnDetectionsReceived?.Invoke(response);
    }
}
