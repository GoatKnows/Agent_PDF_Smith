import streamlit as st
from PIL import Image, ImageFile

# Disable decompression bomb protection or increase limit
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_dpi(image):
    """Extract DPI from image metadata."""
    dpi = image.info.get("dpi")
    if dpi:
        return dpi[0], dpi[1]
    return None

def calculate_physical_size(pixel_width, pixel_height, dpi):
    """Calculate physical size in mm based on DPI and pixel dimensions."""
    mm_per_inch = 25.4
    width_mm = (pixel_width / dpi) * mm_per_inch
    height_mm = (pixel_height / dpi) * mm_per_inch
    return width_mm, height_mm

def calculate_dpi(pixel_width, pixel_height, width_mm, height_mm):
    """Calculate DPI from pixel dimensions and physical size in mm."""
    dpi_x = pixel_width / (width_mm / 25.4)
    dpi_y = pixel_height / (height_mm / 25.4)
    return dpi_x, dpi_y

def is_print_ready(dpi_x, dpi_y, min_dpi=300):
    """Check if DPI meets the print-ready threshold."""
    return dpi_x > min_dpi and dpi_y > min_dpi

# Streamlit App UI
st.title("Automatic Pixel-to-MM Converter")
st.write("Upload an image and select artwork specifications to calculate DPI and physical dimensions.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file:
    try:
        # Open the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")  # Removed use_container_width

        # Extract image dimensions
        pixel_width, pixel_height = image.size
        st.write(f"**Image Dimensions:** {pixel_width} x {pixel_height} pixels")

        # Extract DPI from metadata or assume default DPI
        metadata_dpi = extract_dpi(image)
        if metadata_dpi:
            dpi_x, dpi_y = metadata_dpi
            st.success(f"**DPI from Metadata:** {dpi_x} x {dpi_y} DPI")
        else:
            st.warning("No DPI metadata found in the image.")

        # Display prompt for setting print size
        st.info("Set print size in mm to see DPI")

        # Artwork Specifications Buttons
        st.write("### Select Artwork Specifications")
        if st.button("300mm H x 1980mm W"):
            selected_width_mm = 1980
            selected_height_mm = 300
        elif st.button("2414mm H x 980mm W"):
            selected_width_mm = 980
            selected_height_mm = 2414
        elif st.button("2480mm H x 1960mm W"):
            selected_width_mm = 1960
            selected_height_mm = 2480
        else:
            st.write("### Enter Custom Dimensions")
            selected_width_mm = st.number_input("Enter width in mm:", min_value=1, step=1)
            selected_height_mm = st.number_input("Enter height in mm:", min_value=1, step=1)

        if selected_width_mm and selected_height_mm:
            # Recalculate DPI based on selected dimensions
            dpi_x_manual, dpi_y_manual = calculate_dpi(pixel_width, pixel_height, selected_width_mm, selected_height_mm)
            lowest_dpi = min(dpi_x_manual, dpi_y_manual)

            # Display the lowest DPI value
            st.write(f"**Lowest Recalculated DPI:** {lowest_dpi:.2f}")

            # Check print readiness
            if is_print_ready(dpi_x_manual, dpi_y_manual):
                st.success("The image is print-ready!")
            else:
                st.warning("The image is not print-ready. Consider using a higher-resolution image.")

        else:
            st.info("Please enter both width and height to continue.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
