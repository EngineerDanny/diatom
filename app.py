import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io

def preprocess_for_ml_with_color(image, target_size=None):
    """
    Enhanced preprocessing pipeline preserving color information.
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 1. Convert BGR to RGB if needed
    if len(image.shape) == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image
    else:
        img_rgb = np.stack([image] * 3, axis=-1)
    
    # 2. Resize only if target_size is specified
    if target_size is not None:
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = img_rgb
    
    # 3. Enhance each channel separately using CLAHE
    enhanced = np.zeros_like(resized, dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for i in range(3):
        enhanced[:,:,i] = clahe.apply((resized[:,:,i]).astype(np.uint8))
    
    # 4. Normalize to [-1, 1] range
    normalized = (enhanced.astype(float) - 127.5) / 127.5
    
    return normalized

def detect_diatoms(preprocessed_image, confidence_threshold=0.5):
    """Detect potential diatoms in the image with validation checks"""
    # Convert back to uint8 for OpenCV
    image_uint8 = ((preprocessed_image + 1) * 127.5).astype(np.uint8)
    
    # Validate image content
    if image_uint8.mean() < 10 or image_uint8.mean() > 245:
        raise ValueError("Image appears to be mostly empty or oversaturated")
        
    # Convert to grayscale
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    detections = []
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area > 100 and area < 5000:  # Adjust these thresholds
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 1.0,  # Placeholder
                'area': area
            })
    
    return detections

def draw_detections(image, detections):
    """Draw bounding boxes on the image"""
    # Convert image to uint8 for OpenCV drawing
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)
    
    img_with_boxes = img_uint8.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img_with_boxes

def normalize_for_display(image):
    """Normalize image to [0, 1] range for display"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Handle [-1, 1] range
        if image.min() < 0:
            image = (image + 1) / 2
        # Handle other ranges
        image = np.clip(image, 0, 1)
    else:
        # Handle uint8
        image = image.astype(float) / 255
    return image

def main():
    st.set_page_config(page_title="Diatom Detection", layout="wide")
    
    # Add custom CSS for styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
            margin: 0 auto;
            padding: 1rem;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }
        /* Improved image container styles */
        .block-container {
            max-width: 100%;
            padding-left: 0;
            padding-right: 0;
        }
        .stImage {
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            max-height: 70vh;
        }
        .stImage > img {
            max-width: 100%;
            max-height: 70vh;
            object-fit: contain;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔬 Diatom Detection")
    st.write("Upload microscopy images for automatic diatom detection")
    
    with st.expander("ℹ️ About this app"):
        st.write("""
            This app performs automatic detection of diatoms in microscopy images.
            It supports various image formats including TIFF, JPEG, and PNG.
            The detection algorithm uses advanced image processing techniques to identify potential diatoms.
        """)
    
    # File uploader with supported formats
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['tif', 'tiff', 'jpg', 'jpeg', 'png'],
        help="Upload microscopy images in TIFF, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Validate that this is likely a microscopy image
            if len(image_array.shape) == 3 and image_array.shape[2] > 3:
                st.error("This appears to be a special format image, not a standard microscopy image.")
                return
                
            # Add warning for non-microscopy images
            if image_array.shape[0] < 100 or image_array.shape[1] < 100:
                st.warning("Image dimensions are unusually small for microscopy. Results may not be accurate.")
            
            # Create two columns for original and processed images
            col1, col2 = st.columns(2)
            
            # Create a container div for the images with fixed dimensions
            st.markdown("""
                <style>
                .image-container {
                    width: 100%;
                    max-width: 600px;
                    margin: 0 auto;
                }
                .image-container img {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
                </style>
            """, unsafe_allow_html=True)

            with col1:
                st.subheader("Original Image")
                # Normalize original image for display
                orig_normalized = normalize_for_display(image_array)
                with st.container():
                    st.image(orig_normalized, use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                with st.spinner("Processing image..."):
                    # Preprocess image
                    processed_image = preprocess_for_ml_with_color(image_array)
                    # Detect diatoms
                    detections = detect_diatoms(processed_image)
                    # Normalize processed image for display
                    processed_vis = normalize_for_display(processed_image)
                    # Draw detections
                    result_image = draw_detections(processed_vis, detections)
                    # Final normalization for display
                    result_normalized = normalize_for_display(result_image)
                    with st.container():
                        st.image(result_normalized, use_column_width=True)
            
            # Show detection results
            st.subheader("Detection Results")
            st.write(f"Found {len(detections)} potential diatoms")
            
            if len(detections) > 0:
                # Show area statistics
                areas = [d['area'] for d in detections]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Minimum Area", f"{min(areas):.1f} px²")
                with col2:
                    st.metric("Maximum Area", f"{max(areas):.1f} px²")
                with col3:
                    st.metric("Average Area", f"{np.mean(areas):.1f} px²")
                
                # Option to download results
                results_df = pd.DataFrame(detections)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Detection Results",
                    csv,
                    "diatom_detections.csv",
                    "text/csv",
                    key='download-csv'
                )
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please make sure you uploaded a valid image file.")

if __name__ == "__main__":
    main()