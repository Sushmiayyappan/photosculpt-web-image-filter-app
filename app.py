import streamlit as st
import cv2
import numpy as np
import io # Required for handling image bytes for download

# --- Configuration Constants ---
MAX_DISPLAY_WIDTH = 600  # Max width for each displayed image in the columns
MAX_DOWNLOAD_WIDTH = 1000 # Max width for downloadable images (can be slightly larger than display)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="General Image Filter App",
    page_icon="🎨",
    layout="wide", # Use a wide layout for better display of images
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Helper Function for Image Resizing (maintains aspect ratio) ---
def resize_image(image, max_dim_px):
    h, w = image.shape[:2]
    if max(h, w) > max_dim_px:
        if w > h:
            new_w = max_dim_px
            new_h = int(h * (new_w / w))
        else:
            new_h = max_dim_px
            new_w = int(w * (new_h / h))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# --- Helper function for displaying images in Streamlit ---
def display_image_streamlit(image_to_display, channels="BGR", caption="", col=None):
    # Resize for display
    resized_for_display = resize_image(image_to_display, MAX_DISPLAY_WIDTH)
    if col: # If a column object is provided, display within that column
        col.image(resized_for_display, channels=channels, use_column_width=False, caption=caption)
    else: # Otherwise, display normally (e.g., for the placeholder)
        st.image(resized_for_display, channels=channels, use_column_width=False, caption=caption)


# --- Sidebar Reset Button ---
# Initialize session state for the uploaded file if not already set
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def reset_app():
    st.session_state.uploaded_file = None
    st.rerun() # Rerun the app from the top to reset all widgets

st.sidebar.button("🔄 Reset App", on_click=reset_app)


# --- App Title and Description ---
st.title("🎨 General Image Filter App")
st.write("Upload an image, apply various filters, and download the result!")

# --- Image Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")

# --- Conditional Logic: Only proceed if an image is uploaded ---
if uploaded_file is not None:
    # Store uploaded file in session state to persist after reset
    st.session_state.uploaded_file = uploaded_file

    # --- Convert uploaded file to OpenCV image format ---
    # Read the uploaded file as a byte array
    image_bytes = np.asarray(bytearray(st.session_state.uploaded_file.read()), dtype=np.uint8)
    # Decode the byte array into an OpenCV image (NumPy array)
    # cv2.IMREAD_COLOR ensures it's read as a 3-channel color image
    original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # --- Sidebar for Filter Options ---
    st.sidebar.header("Filter Options")

    # Use a selectbox (dropdown) for selecting the filter type
    filter_type = st.sidebar.selectbox(
        "Select a Filter:",
        ("No Filter", "Gaussian Blur", "Canny Edge Detection", "Black and White", "Quality Adjustment")
    )

    # Make a copy of the original image to process, to avoid modifying the original
    processed_image = original_image.copy()

    # --- Apply Filters based on Selection ---

    if filter_type == "Gaussian Blur":
        st.sidebar.subheader("Gaussian Blur Parameters")
        kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 5, step=2, help="Larger kernel = more blur")
        sigma_x = st.sidebar.slider("Sigma X", 0.0, 10.0, 0.0, step=0.1, help="Standard deviation in X direction")
        processed_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigma_x)

    elif filter_type == "Canny Edge Detection":
        st.sidebar.subheader("Canny Edge Parameters")
        threshold1 = st.sidebar.slider("Threshold 1 (Min Value)", 0, 255, 100, help="Lower threshold for hysteresis")
        threshold2 = st.sidebar.slider("Threshold 2 (Max Value)", 0, 255, 200, help="Upper threshold for hysteresis")
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.Canny(gray_image, threshold1, threshold2)

    elif filter_type == "Black and White":
        st.sidebar.write("Applies a simple grayscale conversion.")
        processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    elif filter_type == "Quality Adjustment":
        st.sidebar.subheader("JPEG Quality Parameters")
        jpeg_quality = st.sidebar.slider(
            "JPEG Quality (0-100)",
            0, 100, 95, # Min, Max, Default
            help="Lower quality reduces file size but introduces more compression artifacts."
        )
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        result, encoded_image_bytes_arr = cv2.imencode('.jpg', original_image, encode_param)

        if result:
            processed_image = cv2.imdecode(encoded_image_bytes_arr, cv2.IMREAD_COLOR)
        else:
            st.error("Failed to apply quality adjustment.")
            processed_image = original_image.copy() # Fallback to original

    elif filter_type == "No Filter":
        # processed_image is already original_image.copy()
        pass # Do nothing, effectively showing original image

    # --- Display Images in Comparison Mode (Side-by-Side) ---
    st.subheader("Image Comparison")
    # Create two columns for original and processed images
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original Image")
        display_image_streamlit(original_image, channels="BGR", caption=f"Original ({original_image.shape[1]}x{original_image.shape[0]})", col=col1)

    with col2:
        st.write(f"### Processed Image: {filter_type}")
        if len(processed_image.shape) == 2: # Check if grayscale (2D)
            display_image_streamlit(processed_image, channels="GRAY", caption=f"Processed ({processed_image.shape[1]}x{processed_image.shape[0]})", col=col2)
        else: # Color (3D)
            display_image_streamlit(processed_image, channels="BGR", caption=f"Processed ({processed_image.shape[1]}x{processed_image.shape[0]})", col=col2)


    # --- Download Option ---
    st.markdown("---") # Separator for visual clarity
    st.subheader("Download Processed Image")
    download_format = st.selectbox(
        "Select download format:",
        ("jpeg", "png"), # Options for download format
        key="download_format_select" # Unique key to avoid warning if other selectboxes exist
    )

    download_filename = f"filtered_image.{download_format}"
    mime_type = f"image/{download_format}"

    # Resize processed image for download (if it's larger than MAX_DOWNLOAD_WIDTH)
    image_for_download = resize_image(processed_image, MAX_DOWNLOAD_WIDTH)

    # Encode the image for download
    if len(image_for_download.shape) == 2: # Grayscale
        if download_format == "jpeg":
            result, encoded_img_bytes = cv2.imencode(f'.{download_format}', image_for_download, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else: # PNG
            result, encoded_img_bytes = cv2.imencode(f'.{download_format}', image_for_download)
    else: # Color image
        if download_format == "jpeg":
            result, encoded_img_bytes = cv2.imencode(f'.{download_format}', image_for_download, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else: # PNG
            result, encoded_img_bytes = cv2.imencode(f'.{download_format}', image_for_download)

    if result:
        st.download_button(
            label=f"Download Image as .{download_format}",
            data=io.BytesIO(encoded_img_bytes.tobytes()), # Convert numpy array of bytes to actual bytes
            file_name=download_filename,
            mime=mime_type
        )
    else:
        st.error("Could not prepare image for download.")

# --- Initial State: When no image is uploaded ---
else:
    st.info("⬆ Please upload an image using the button above to apply filters.")
    # Placeholder image, also resized to a medium size
    st.image("https://via.placeholder.com/800x500?text=Upload+Your+Image+Here",
             caption="Waiting for an image...",
             use_column_width=True)