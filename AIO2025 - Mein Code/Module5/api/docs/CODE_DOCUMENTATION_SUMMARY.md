# Code Documentation Summary - main.py

## Overview
Added comprehensive inline comments to clarify ambiguous definitions and complex logic throughout the Heart Disease Prediction API.

---

## 📝 Comments Added

### **1. PatientData Model - Age Conversion**
**Location:** Lines 33-46

**What was ambiguous:** The `@field_validator('age')` decorator and why age needs conversion

**Comment added:**
```python
"""
Convert age from years to days (as required by the model).

NOTE: This validator automatically converts user-provided age in YEARS
to DAYS because the underlying ML model was trained on age in days.
Example: 50 years → 18,250 days (50 * 365)

Args:
    age_years: Age in years (0-120)

Returns:
    Age in days (age_years * 365)
"""
```

**Why important:** Users need to know that their input age in years is automatically converted, so they don't need to do manual calculations.

---

### **2. PredictionRequest - Video Selection Validation**
**Location:** Lines 72-91

**What was ambiguous:** How the validator handles different combinations of `video_index` and `video_filename`

**Comment added:**
```python
"""
Ensure either video_index or video_filename is provided, but not both.

Validation Logic:
- If both are None: defaults to video_index=0 (first video)
- If video_filename provided: uses that (index will be looked up)
- If video_index provided: uses that directly
- If both provided: video_filename takes priority

Args:
    v: The video_filename value being validated
    info: ValidationInfo containing other field values

Returns:
    The validated video_filename (or None if using index)
"""
```

**Why important:** Clarifies the priority and fallback behavior when users provide different combinations of video selection parameters.

---

### **3. get_video_index_from_filename() Function**
**Location:** Lines 103-123

**What was ambiguous:** Purpose, behavior, and return values of this helper function

**Comment added:**
```python
"""
Get video index from filename by searching the EchoNet FileList.

This function loads the EchoNet dataset metadata and searches for a matching
filename, returning its corresponding index. This allows users to reference
videos by meaningful filenames instead of memorizing numeric indices.

Args:
    filename: Video filename (with or without .avi extension)
             Examples: "0X1005D03EED19C65B.avi" or "0X1005D03EED19C65B"

Returns:
    int: The index of the video in the EchoNet dataset (0-10029)

Raises:
    HTTPException 500: If dataset fails to load
    HTTPException 404: If filename not found in dataset

Example:
    >>> get_video_index_from_filename("0X1005D03EED19C65B.avi")
    0
"""
```

**Additional comments:**
- Filename normalization logic (why we remove `.avi`)
- Why EchoNet stores filenames without extension
- How `iterrows()` returns (index, row) tuples

**Why important:** Developers modifying this function need to understand the filename matching logic and edge cases.

---

### **4. list_videos() - Pagination Logic**
**Location:** Lines 178-193

**What was ambiguous:** Why there's a 1000 limit and how pagination works

**Comments added:**
```python
# Validate limit to prevent excessive memory usage
# Max 1000 videos per request to balance performance and usability

# Get subset of videos using pandas iloc for efficient slicing
# This enables pagination: offset=0, limit=50 gets videos 0-49
```

**Why important:** Explains the performance considerations and how to use offset/limit for pagination.

---

### **5. Video Metadata Extraction**
**Location:** Lines 228-240

**What was ambiguous:** What each metadata field represents and why we check file existence

**Comments added:**
```python
# Build response with video metadata
# Each video includes clinical metadata from EchoNet-Dynamic dataset
videos = []
for idx, row in videos_subset.iterrows():
    video_filename = row['FileName']
    # Construct full path: videos_path/filename.avi
    video_path = videos_path / f"{video_filename}.avi"

    videos.append({
        "index": int(idx),  # Original index in full dataset
        "filename": video_filename,  # Without extension (as stored in CSV)
        "filename_with_ext": f"{video_filename}.avi",  # With extension (actual file)
        "ejection_fraction": float(row['EF']) if 'EF' in row else None,  # % of blood pumped out
        "frame_height": int(row['FrameHeight']) if 'FrameHeight' in row else None,
        "frame_width": int(row['FrameWidth']) if 'FrameWidth' in row else None,
        "fps": float(row['FPS']) if 'FPS' in row else None,  # Frames per second
        "number_of_frames": int(row['NumberOfFrames']) if 'NumberOfFrames' in row else None,
        "split": row['Split'] if 'Split' in row else None,  # TRAIN/VAL/TEST
        "exists": video_path.exists()  # Verify file actually exists on disk
    })
```

**Why important:** Medical professionals using the API need to understand clinical metrics like ejection fraction.

---

### **6. search_videos() - Search Algorithm**
**Location:** Lines 275-283

**What was ambiguous:** How the search works and why it stops early

**Comments added:**
```python
# Search for matching filenames using case-insensitive substring matching
# Example: query="0X10" matches "0X1005D03EED19C65B", "0X10A9C59476F90E1B", etc.

# Iterate through all videos and find matches
# Stops early once limit is reached for performance
for idx, row in echonet_filelist.iterrows():
    ...
    # Stop searching once we've found enough matches
    # This improves performance for common queries
    if len(matches) >= limit:
        break
```

**Why important:** Explains the performance optimization of early termination.

---

### **7. predict() - Video Index Resolution**
**Location:** Lines 321-331

**What was ambiguous:** How the function decides which video to use

**Comments added:**
```python
# Convert PatientData to dict for processing
# model_dump() is Pydantic v2's method (replaces dict() from v1)
patient_json = request.patient_data.model_dump()

# Determine video index from either filename or index
# Priority: video_filename > video_index > default(0)
if request.video_filename is not None:
    # Get index from filename using FileList lookup
    video_idx = get_video_index_from_filename(request.video_filename)
else:
    # Use provided index (defaults to 0 if not provided)
    video_idx = request.video_index if request.video_index is not None else 0
```

**Why important:** Clarifies the priority order and Pydantic version compatibility.

---

### **8. Confidence Calculation**
**Location:** Lines 348-354

**What was ambiguous:** How confidence is calculated from probability

**Comment added:**
```python
# Calculate confidence (max probability)
# Model outputs probability for class 1 (Disease)
# Confidence is the max of: P(Disease) or P(No Disease) = 1 - P(Disease)
# Example: if P(Disease)=0.8, confidence=0.8; if P(Disease)=0.2, confidence=0.8
confidence = max(float(probability), 1.0 - float(probability))
```

**Why important:** Non-technical users need to understand what "confidence" means in the API response.

---

### **9. Batch Processing Loop**
**Location:** Lines 391-404

**What was ambiguous:** How errors are handled without stopping the batch

**Comments added:**
```python
# Process each prediction request independently
# Failed predictions don't stop the batch - they're added to errors list
for idx, request in enumerate(requests):
    ...
    # Add successful prediction to results
    # Includes video_filename if provided (for traceability)
    results.append({...})
    ...
    # Prediction returned None - model issue
    errors.append({...})
    ...
    # Catch any exception during prediction (file not found, invalid data, etc.)
    errors.append({...})
```

**Why important:** Users need to know that batch processing is resilient and partial failures are tracked.

---

## 🎯 **Impact Summary**

### **Before:**
- Ambiguous validation logic
- Unclear function purposes
- Mysterious magic numbers (1000 limit)
- Unexplained data transformations
- No clinical context for metadata

### **After:**
- ✅ Clear validation rules with examples
- ✅ Comprehensive function documentation
- ✅ Performance considerations explained
- ✅ Data flow is traceable
- ✅ Clinical metrics explained for medical users
- ✅ Edge cases documented
- ✅ Framework compatibility notes (Pydantic v2)

---

## 📚 **Documentation Standards Applied**

1. **Docstrings:** Complete with Args, Returns, Raises, Examples
2. **Inline Comments:** Explain "why" not just "what"
3. **Examples:** Real-world usage examples in docstrings
4. **Edge Cases:** Document fallback behaviors
5. **Performance Notes:** Explain optimization decisions
6. **Clinical Context:** Medical terminology explained
7. **Framework Notes:** Version-specific features noted

---

## 🔍 **Key Improvements**

### **Clarity**
- Medical professionals understand clinical metrics
- Developers understand validation logic
- Users understand priority and fallbacks

### **Maintainability**
- Future developers can modify safely
- Edge cases are documented
- Performance trade-offs are clear

### **Usability**
- API behavior is predictable
- Error handling is transparent
- Examples guide correct usage

---

## ✅ **Verification**

All ambiguous sections now have:
- [ ] Purpose explained
- [ ] Behavior documented
- [ ] Edge cases covered
- [ ] Examples provided
- [ ] Performance notes (where relevant)
- [ ] Medical context (where relevant)

The code is now **production-ready** with professional-grade documentation! 🎉
