# FastAPI Timelapse Video Generation Service - Design Document

**Date:** 2026-02-08
**Purpose:** Web service that downloads images from Google Drive, stabilizes them, and generates a timelapse video

---

## Overview

A FastAPI web service that:
1. Accepts a public Google Drive folder link + reference image name + webhook URL
2. Downloads all images from the folder
3. Sorts images by creation date
4. Chain-compares and corrects each image against its predecessor (reference sets baseline)
5. Generates a stabilized timelapse video (30fps, 1080p, MP4)
6. Notifies user via webhook with a download URL

---

## Architecture

```
Video_generation_pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, endpoint definitions
│   ├── models.py                # Pydantic request/response models
│   ├── routes/
│   │   ├── __init__.py
│   │   └── jobs.py              # Job endpoints (POST, GET, download)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gdrive.py            # Google Drive download (public links)
│   │   ├── pipeline.py          # Main processing pipeline
│   │   ├── comparator.py        # Image comparison (from image_comparator.py)
│   │   ├── corrector.py         # Image correction (from image_corrector.py)
│   │   ├── aligner.py           # Crop/gray fill (from image_aligner.py)
│   │   └── video.py             # Video generation (new)
│   └── worker.py                # Background task processing
├── feature_detector.py          # Existing CLI tool (unchanged)
├── image_comparator.py          # Existing CLI tool (unchanged)
├── image_corrector.py           # Existing CLI tool (unchanged)
├── image_aligner.py             # Existing CLI tool (unchanged)
├── requirements.txt             # Updated with FastAPI deps
└── docs/plans/
```

---

## API Endpoints

### 1. `POST /jobs` - Create Timelapse Job

**Request Body:**
```json
{
    "drive_link": "https://drive.google.com/drive/folders/ABC123",
    "reference_image": "20250521170005.JPG",
    "webhook_url": "https://example.com/webhook/callback"
}
```

**Response (202 Accepted):**
```json
{
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "queued",
    "message": "Job created. You will be notified at webhook_url when complete."
}
```

**Validation:**
- `drive_link` must be a valid Google Drive folder URL
- `reference_image` must be a filename (validated after download)
- `webhook_url` must be a valid HTTP/HTTPS URL

---

### 2. `GET /jobs/{job_id}` - Check Job Status

**Response:**
```json
{
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "processing",
    "progress": {
        "step": "correcting",
        "current_image": 5,
        "total_images": 20,
        "percentage": 25
    },
    "created_at": "2026-02-08T10:30:00Z"
}
```

**Status Values:**
| Status | Meaning |
|--------|---------|
| `queued` | Job received, waiting to start |
| `downloading` | Downloading images from Google Drive |
| `processing` | Comparing and correcting images |
| `generating_video` | Creating timelapse MP4 |
| `completed` | Done, video ready for download |
| `failed` | Error occurred, see error message |

---

### 3. `GET /jobs/{job_id}/download` - Download Video

**Response:** Video file stream (MP4)

**Headers:**
```
Content-Type: video/mp4
Content-Disposition: attachment; filename="timelapse_{job_id}.mp4"
```

**Errors:**
- `404` if job_id not found
- `400` if job not yet completed

---

## Processing Pipeline

### Chain Comparison Flow

```
Reference Image (user selected)
       |
       v
   [Compare with Image 1]
       |
       ├── Feature match < 15%? → FAIL JOB (image doesn't belong)
       |
       ├── Crop detected? → Fill from reference (image_aligner)
       ├── Misaligned? → Shift to align (affine translation)
       ├── Color shift? → Match color balance
       ├── Brightness diff? → Histogram matching
       ├── Sharpness diff? → Sharpen or blur to match
       |
       v
   Modified Image 1
       |
       v
   [Compare Modified Image 1 with Image 2]
       |
       ├── Same correction checks...
       |
       v
   Modified Image 2
       |
       v
   [Compare Modified Image 2 with Image 3]
       |
       ... continues for all images ...
       |
       v
   All Modified Images → Video Generator → timelapse.mp4
```

### Correction Order (per image pair)

Applied in this sequence to avoid conflicts:

1. **Crop detection & fill** - Restore missing gray areas from predecessor
2. **Alignment** - Simple translation (dx, dy shift) using median of best 30% feature matches
3. **Color balance** - Scale R/G/B channel means to match predecessor
4. **Brightness & contrast** - Histogram matching via CDF lookup tables
5. **Sharpness** - Unsharp mask or Gaussian blur to match predecessor

### Failure Condition

If any image has **< 15% feature match coverage** with its predecessor, the job fails immediately with:

```json
{
    "status": "failed",
    "error": "Image '20250610_selfie.JPG' does not match the sequence. Only 8.2% feature match (minimum 15% required). Remove this image and try again."
}
```

---

## Google Drive Download

### How Public Folder Links Work

1. User shares folder with "Anyone with the link can view"
2. Folder URL format: `https://drive.google.com/drive/folders/{folder_id}`
3. Extract `folder_id` from URL
4. Use Google Drive API (no auth needed for public folders) to list files
5. Download each image file

### Implementation

```
Input: https://drive.google.com/drive/folders/ABC123
          |
          v
Extract folder_id = "ABC123"
          |
          v
GET https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&key={API_KEY}
          |
          v
For each file: Download to /tmp/jobs/{job_id}/originals/
          |
          v
Filter: Keep only image files (JPG, PNG, BMP, TIFF)
```

**Note:** Requires a Google API key (free, no OAuth). Set via environment variable `GOOGLE_API_KEY`.

---

## Webhook Notification

### On Success

```json
POST {webhook_url}

{
    "job_id": "a1b2c3d4",
    "status": "completed",
    "download_url": "http://server:8000/jobs/a1b2c3d4/download",
    "summary": {
        "total_images": 20,
        "images_corrected": 18,
        "video_duration_seconds": 0.67,
        "video_resolution": "1920x1080",
        "video_fps": 30
    }
}
```

### On Failure

```json
POST {webhook_url}

{
    "job_id": "a1b2c3d4",
    "status": "failed",
    "error": "Image 'selfie.JPG' does not match the sequence."
}
```

---

## File Storage

### Directory Structure Per Job

```
/tmp/jobs/{job_id}/
├── originals/              # Downloaded from Google Drive
│   ├── 20250521170005.JPG
│   ├── 20250522080009.JPG
│   └── ...
├── corrected/              # After chain comparison & correction
│   ├── 0001_20250521170005.JPG
│   ├── 0002_20250522080009.JPG
│   └── ...
├── report.json             # Processing report
└── output/
    └── timelapse.mp4       # Final video
```

### Cleanup

- Job files are deleted **24 hours** after completion
- A background cleanup task runs periodically to remove expired jobs

---

## Video Generation

### Settings (Fixed)

| Setting | Value |
|---------|-------|
| Format | MP4 |
| Codec | mp4v |
| FPS | 30 |
| Resolution | 1920x1080 (resize all frames) |

### Process

1. Load all corrected images in order
2. Resize each to 1920x1080 (maintain aspect ratio, pad if needed)
3. Write frames sequentially using `cv2.VideoWriter`
4. Release and finalize

---

## Data Models

### JobRequest

```python
class JobRequest(BaseModel):
    drive_link: str        # Google Drive public folder URL
    reference_image: str   # Filename of reference image
    webhook_url: str       # URL to notify when done
```

### JobStatus

```python
class JobStatus(BaseModel):
    job_id: str
    status: str            # queued/downloading/processing/generating_video/completed/failed
    progress: dict | None  # step, current_image, total_images, percentage
    created_at: datetime
    error: str | None
```

### WebhookPayload

```python
class WebhookPayload(BaseModel):
    job_id: str
    status: str
    download_url: str | None
    error: str | None
    summary: dict | None
```

---

## Dependencies

### New (add to requirements.txt)

```
fastapi>=0.100.0
uvicorn>=0.20.0
httpx>=0.24.0              # For webhook HTTP calls
python-multipart>=0.0.5    # For file handling
```

### Existing (keep)

```
opencv-python>=4.5.0
numpy>=1.20.0
```

---

## Error Handling

| Error | HTTP Code | Action |
|-------|-----------|--------|
| Invalid Drive URL | 400 | Reject with message |
| Drive folder not accessible | 400 | "Folder is not publicly shared" |
| Reference image not found | 400 | "Reference image '{name}' not found in folder" |
| No images in folder | 400 | "No image files found in folder" |
| Less than 2 images | 400 | "Need at least 2 images for timelapse" |
| Image doesn't match sequence | Job fails | Webhook with error, name the bad image |
| Download timeout | Job fails | Webhook with error |
| Video generation fails | Job fails | Webhook with error |

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | Yes | Google Drive API key for public folder access |
| `HOST` | No | Server host (default: 0.0.0.0) |
| `PORT` | No | Server port (default: 8000) |
| `JOB_CLEANUP_HOURS` | No | Hours before job files are deleted (default: 24) |

---

## Implementation Order

1. **Project setup** - FastAPI app skeleton, models, requirements
2. **Google Drive service** - Download images from public folder link
3. **Pipeline service** - Chain comparison & correction (reuse existing code)
4. **Video service** - Generate MP4 from corrected images
5. **Background worker** - Async job processing
6. **Webhook notification** - POST results to callback URL
7. **Job endpoints** - POST /jobs, GET /jobs/{id}, GET /jobs/{id}/download
8. **Cleanup task** - Delete expired job files
9. **Testing** - End-to-end test with real Google Drive folder
