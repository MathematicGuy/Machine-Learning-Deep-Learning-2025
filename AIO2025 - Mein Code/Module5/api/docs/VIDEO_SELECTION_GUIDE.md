# Video Selection Guide

## 🎥 New Video Selection Features

You can now select videos in **3 ways**:

### **1. By Video Index (Original Method)**
```json
{
  "patient_data": { ... },
  "video_index": 2
}
```

### **2. By Video Filename**
```json
{
  "patient_data": { ... },
  "video_filename": "0X1005D03EED19C65B.avi"
}
```

### **3. Browse Available Videos First**

#### **List All Videos (with pagination)**
```bash
GET http://localhost:8000/videos?limit=50&offset=0
```

**Response:**
```json
{
  "videos": [
    {
      "index": 0,
      "filename": "0X1005D03EED19C65B",
      "filename_with_ext": "0X1005D03EED19C65B.avi",
      "ejection_fraction": 35.0,
      "frame_height": 112,
      "frame_width": 112,
      "fps": 50.0,
      "number_of_frames": 163,
      "split": "TRAIN",
      "exists": true
    },
    ...
  ],
  "total_videos": 10030,
  "offset": 0,
  "limit": 50,
  "returned": 50,
  "videos_path": "C:\\Users\\APC\\.cache\\kagglehub\\datasets\\..."
}
```

#### **Search for Specific Videos**
```bash
GET http://localhost:8000/videos/search?query=0X1005&limit=10
```

**Response:**
```json
{
  "query": "0X1005",
  "matches": [
    {
      "index": 0,
      "filename": "0X1005D03EED19C65B",
      "filename_with_ext": "0X1005D03EED19C65B.avi",
      "ejection_fraction": 35.0,
      "exists": true
    },
    ...
  ],
  "total_matches": 5,
  "limit": 10
}
```

---

## 📝 **Usage Examples**

### **Python with requests**

```python
import requests

API_URL = "http://localhost:8000"

# 1. List available videos
response = requests.get(f"{API_URL}/videos?limit=10")
videos = response.json()["videos"]

print("Available videos:")
for video in videos[:5]:
    print(f"  [{video['index']}] {video['filename_with_ext']} - EF: {video['ejection_fraction']}")

# 2. Search for a specific video
response = requests.get(f"{API_URL}/videos/search?query=0X1005")
matches = response.json()["matches"]
print(f"\nFound {len(matches)} matching videos")

# 3. Make prediction using video INDEX
patient_data = {
    "id": 12345,
    "age": 50,  # in years
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cardio": 0
}

# Using video index
response = requests.post(
    f"{API_URL}/predict",
    json={
        "patient_data": patient_data,
        "video_index": 2
    }
)
print(f"\nPrediction with index: {response.json()['prediction_label']}")

# 4. Make prediction using video FILENAME
response = requests.post(
    f"{API_URL}/predict",
    json={
        "patient_data": patient_data,
        "video_filename": "0X1005D03EED19C65B.avi"
    }
)
print(f"Prediction with filename: {response.json()['prediction_label']}")
```

### **Interactive Swagger UI**

1. Start the server:
   ```bash
   cd "d:\CODE\Machine-Learning-Deep-Learning-2025\AIO2025 - Mein Code\Module5\api"
   python main.py
   ```

2. Open browser: `http://localhost:8000/docs`

3. Try the new endpoints:
   - **GET /videos** - Browse available videos
   - **GET /videos/search** - Search for videos
   - **POST /predict** - Make predictions with video_filename

---

## 🔍 **Video Selection Workflow**

### **Step-by-Step Guide**

1. **Browse Videos**
   ```bash
   curl "http://localhost:8000/videos?limit=20"
   ```

2. **Find Interesting Video**
   ```json
   {
     "index": 15,
     "filename": "0X10A9C59476F90E1B",
     "filename_with_ext": "0X10A9C59476F90E1B.avi",
     "ejection_fraction": 45.2,
     "exists": true
   }
   ```

3. **Use in Prediction** (Choose one method)
   ```json
   // Method A: Use index
   {
     "patient_data": { ... },
     "video_index": 15
   }

   // Method B: Use filename
   {
     "patient_data": { ... },
     "video_filename": "0X10A9C59476F90E1B.avi"
   }
   ```

---

## 🎯 **Request Model Changes**

### **New Optional Fields**

```python
class PredictionRequest:
    patient_data: PatientData          # Required
    video_index: Optional[int]         # Optional (use this OR video_filename)
    video_filename: Optional[str]      # Optional (use this OR video_index)
```

### **Validation Rules**

- ✅ **Only `video_index`**: Uses that index
- ✅ **Only `video_filename`**: Looks up index from filename
- ✅ **Neither provided**: Defaults to `video_index=0`
- ❌ **Both provided**: `video_filename` takes priority

---

## 📊 **Example Request/Response**

### **Request with Filename**
```json
{
  "patient_data": {
    "id": 12345,
    "age": 50,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
    "cardio": 0
  },
  "video_filename": "0X1005D03EED19C65B.avi"
}
```

### **Response**
```json
{
  "prediction": 0,
  "prediction_label": "No Disease",
  "disease_probability": 0.1234,
  "confidence": 0.8766,
  "patient_id": 12345,
  "video_index": 0,
  "ground_truth": 0,
  "numerical_features_shape": [1, 28],
  "cnn_features_shape": [1, 64],
  "device": "cuda"
}
```

---

## 🛠️ **Technical Details**

### **File Path Location**
```
C:\Users\APC\.cache\kagglehub\datasets\mahnurrahman\echonet-dynamic\versions\1\EchoNet-Dynamic\Videos\
```

### **Video Files**
- Format: `.avi` files
- Total count: 10,030 videos
- Naming: Alphanumeric identifiers (e.g., `0X1005D03EED19C65B.avi`)

### **Filename Resolution**
When you provide a `video_filename`, the API:
1. Loads the EchoNet FileList
2. Removes `.avi` extension for matching
3. Searches for exact filename match
4. Returns the corresponding index
5. Uses that index for prediction

---

## ✅ **Summary**

**New Endpoints:**
- `GET /videos` - List all videos with pagination
- `GET /videos/search` - Search videos by filename pattern

**Enhanced Features:**
- Select videos by filename instead of memorizing indices
- Browse and search available videos
- Automatic filename-to-index conversion
- Flexible video selection in predictions

**Benefits:**
- 🎯 More intuitive video selection
- 🔍 Easy video discovery
- 📊 View video metadata before selection
- 🚀 Production-ready API
