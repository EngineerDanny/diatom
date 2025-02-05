import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import io

def preprocess_for_ml_with_color(image, target_size=(512, 512)):
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
    
    # 2. Resize while preserving aspect ratio
    resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Enhance each channel separately using CLAHE
    enhanced = np.zeros_like(resized, dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for i in range(3):
        enhanced[:,:,i] = clahe.apply((resized[:,:,i]).astype(np.uint8))
    
    # 4. Normalize to [-1, 1] range
    normalized = (enhanced.astype(float) - 127.5) / 127.5
    
    return normalized

def detect_diatoms(preprocessed_image, confidence_threshold=0.5):
    # Convert back to uint8 for OpenCV
    image_uint8 = ((preprocessed_image + 1) * 127.5).astype(np.uint8)
    
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
    img_with_boxes = image.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img_with_boxes

def local_css(file_name):
    """Apply custom CSS"""
    st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #f8f9fa;
        }
        
        /* Custom upload box */
        .upload-box {
            border: 2px dashed #6c757d;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background-color: #ffffff;
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 1rem 0;
        }
        .upload-box:hover {
            border-color: #0d6efd;
            background-color: #f8f9fa;
        }
        
        /* Headers styling */
        h1 {
            color: #1e1e1e;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 3px solid #0d6efd;
        }
        h2 {
            color: #2c3e50;
            font-weight: 600;
            margin-top: 1.5rem;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Results section */
        .results-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Metrics styling */
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            background-color: #0d6efd;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0b5ed7;
            transform: translateY(-2px);
        }
        
        /* Download button */
        .download-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #198754;
            color: white;
            border-radius: 5px;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .download-button:hover {
            background-color: #157347;
        }
        
        /* About section */
        .about-section {
            background-color: #ffffff;
            border-left: 4px solid #0d6efd;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Loading spinner */
        .stSpinner {
            text-align: center;
            color: #0d6efd;
        }
        
        /* Error message */
        .stAlert {
            border-radius: 8px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Diatom Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    local_css("style.css")
    
    # App header
    st.markdown("""
        <h1>🔬 Diatom Detection</h1>
        <p style='text-align: center; color: #6c757d; font-size: 1.2rem;'>
            Advanced microscopy image analysis for diatom detection
        </p>
    """, unsafe_allow_html=True)
    
    # About section with custom styling
    with st.expander("ℹ️ About this application"):
        st.markdown("""
            <div class='about-section'>
                <h3>Welcome to the Diatom Detection Tool</h3>
                <p>This application utilizes advanced image processing techniques to automatically detect and analyze diatoms in microscopy images. Key features include:</p>
                <ul>
                    <li>Support for multiple image formats (TIFF, JPEG, PNG)</li>
                    <li>Real-time image processing and analysis</li>
                    <li>Detailed detection statistics and measurements</li>
                    <li>Downloadable results in CSV format</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your microscopy image",
        type=['tif', 'tiff', 'jpg', 'jpeg', 'png'],
        help="Supported formats: TIFF, JPEG, PNG"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Read and process image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Create two columns for image display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>Original Image</h2>", unsafe_allow_html=True)
                st.image(image_array, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>Processed Image</h2>", unsafe_allow_html=True)
                with st.spinner("Processing image..."):
                    processed_image = preprocess_for_ml_with_color(image_array)
                    detections = detect_diatoms(processed_image)
                    processed_vis = ((processed_image + 1) / 2)
                    result_image = draw_detections(processed_vis, detections)
                    st.image(result_image, use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Results section
            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            st.markdown(f"<h2>Detection Results</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #0d6efd; text-align: center;'>Found {len(detections)} potential diatoms</h3>", unsafe_allow_html=True)
            
            if len(detections) > 0:
                # Statistics display
                areas = [d['area'] for d in detections]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Minimum Area", f"{min(areas):.1f} px²")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Maximum Area", f"{max(areas):.1f} px²")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Average Area", f"{np.mean(areas):.1f} px²")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Download section
                st.markdown("<br>", unsafe_allow_html=True)
                results_df = pd.DataFrame(detections)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Detection Results",
                    csv,
                    "diatom_detections.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.warning("Please make sure you uploaded a valid microscopy image file.")

if __name__ == "__main__":
    main()