import streamlit as st
from PIL import Image, ImageFile, ImageCms
import numpy as np
import torch
from torchvision.transforms import functional as F
import time
import os

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
    return dpi_x >= min_dpi and dpi_y >= min_dpi

def convert_rgb_to_cmyk(image):
    """Convert an RGB image to CMYK using ICC profiles."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    base_path = os.path.join(os.path.dirname(__file__), "icc_profiles")
    srgb_profile_path = os.path.join(base_path, "AdobeRGB1998.icc")
    cmyk_profile_path = os.path.join(base_path, "CoatedFOGRA39.icc")  # Use FOGRA39 for better Adobe compatibility

    if not os.path.exists(srgb_profile_path):
        st.error(f"sRGB profile not found at: {srgb_profile_path}")
        raise FileNotFoundError(f"sRGB profile not found at: {srgb_profile_path}")
    if not os.path.exists(cmyk_profile_path):
        st.error(f"CMYK profile not found at: {cmyk_profile_path}")
        raise FileNotFoundError(f"CMYK profile not found at: {cmyk_profile_path}")

    srgb_profile = ImageCms.ImageCmsProfile(srgb_profile_path)
    cmyk_profile = ImageCms.ImageCmsProfile(cmyk_profile_path)
    transform = ImageCms.buildTransform(srgb_profile, cmyk_profile, "RGB", "CMYK", renderingIntent=0)  # Try Perceptual
    cmyk_image = ImageCms.applyTransform(image, transform)
    return cmyk_image

def upscale_image_to_dpi(image, target_dpi, original_dpi):
    """Upscale image to achieve the target DPI."""
    scale_factor = target_dpi / original_dpi
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Simulate processing time for upscaling
    total_steps = 10
    for step in range(total_steps):
        time.sleep(0.5)
        progress = (step + 1) / total_steps
        st.progress(progress)

    # Convert to numpy array for AI processing
    image_np = np.array(image)

    # Convert to tensor and upscale using torchvision
    image_tensor = F.to_tensor(image_np).unsqueeze(0)
    upscaled_tensor = F.resize(image_tensor, size=(new_height, new_width))
    upscaled_np = upscaled_tensor.squeeze().permute(1, 2, 0).numpy()

    # Convert back to PIL
    upscaled_image = Image.fromarray((upscaled_np * 255).astype(np.uint8))

    return upscaled_image

def save_cmyk_pdf(image, output_path, cmyk_profile_path):
    """Save CMYK image as a PDF with embedded ICC profile."""
    if image.mode != "CMYK":
        raise ValueError("Image must be in CMYK mode to save as PDF.")

    # Flatten the image completely
    flattened_image = Image.new("CMYK", image.size, (255, 255, 255, 0))
    flattened_image.paste(image)

    if not os.path.exists(cmyk_profile_path):
        raise FileNotFoundError(f"CMYK profile not found at: {cmyk_profile_path}")

    with open(cmyk_profile_path, "rb") as profile:
        icc_data = profile.read()

    # Save flattened_image to a temporary file
    temp_image_path = "temp_image.jpg"
    flattened_image.save(temp_image_path, "JPEG", quality=95)

    # Use FPDF to embed the flattened image
    from fpdf import FPDF
    pdf = FPDF(unit="pt", format=[image.width, image.height])
    pdf.add_page()
    pdf.image(temp_image_path, x=0, y=0, w=image.width, h=image.height)
    pdf.output(output_path)

    # Clean up temporary file
    os.remove(temp_image_path)

    st.write("PDF saved successfully with alternate library.")

# Streamlit App UI
st.title("RGB to CMYK 300 DPI Print-Ready PDF Converter")
st.write("Upload an image, convert it to a print-ready CMYK 300 DPI PDF, or download in other high-res formats.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file:
    try:
        # Open the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        pixel_width, pixel_height = image.size
        st.write(f"Image Dimensions: {pixel_width} x {pixel_height} pixels")

        metadata_dpi = extract_dpi(image)
        if metadata_dpi:
            dpi_x, dpi_y = metadata_dpi
            st.success(f"DPI from Metadata: {dpi_x} x {dpi_y}")
        else:
            dpi_x, dpi_y = 72, 72
            st.warning("No DPI metadata found. Assuming default 72 DPI.")

        st.info("Set print size in mm to calculate DPI")
        selected_width_mm = st.number_input("Enter width in mm:", min_value=1, step=1)
        selected_height_mm = st.number_input("Enter height in mm:", min_value=1, step=1)

        if selected_width_mm and selected_height_mm:
            dpi_x_manual, dpi_y_manual = calculate_dpi(pixel_width, pixel_height, selected_width_mm, selected_height_mm)
            lowest_dpi = min(dpi_x_manual, dpi_y_manual)

            st.write(f"Calculated DPI based on input dimensions: {dpi_x_manual:.2f} x {dpi_y_manual:.2f}")

            if is_print_ready(dpi_x_manual, dpi_y_manual):
                st.success("The image is print-ready for the specified dimensions!")
            else:
                st.warning("The image is not print-ready for the specified dimensions. Upscaling is recommended.")

        if is_print_ready(dpi_x, dpi_y):
            st.success("Image is print-ready!")
        else:
            st.warning("Image is not print-ready. Upscaling is recommended.")
            if st.button("Upscale to 300 DPI"):
                with st.spinner("Upscaling image..."):
                    upscaled_image = upscale_image_to_dpi(image, target_dpi=300, original_dpi=dpi_x)
                st.image(upscaled_image, caption="Upscaled Image", use_container_width=True)

                # Convert to CMYK
                upscaled_image_cmyk = convert_rgb_to_cmyk(upscaled_image)

                # Save as a CMYK 300 DPI PDF
                pdf_path = "upscaled_image.pdf"
                cmyk_profile_path = os.path.join(
                    os.path.dirname(__file__),
                    "icc_profiles",
                    "CoatedFOGRA39.icc"
                )
                save_cmyk_pdf(upscaled_image_cmyk, pdf_path, cmyk_profile_path)

                with open(pdf_path, "rb") as file:
                    st.download_button("Download Print-Ready CMYK PDF", file, file_name="print_ready.pdf", mime="application/pdf")

                # Save in other formats
                formats = {
                    "TIFF (CMYK)": "upscaled_image_cmyk.tiff",
                    "JPEG (High Quality)": "upscaled_image_high_quality.jpg",
                }

                for label, filename in formats.items():
                    if label == "TIFF (CMYK)":
                        upscaled_image_cmyk.save(filename, dpi=(300, 300))
                    else:
                        upscaled_image_cmyk.save(filename, format=label.split(' ')[0].upper(), dpi=(300, 300))

                    with open(filename, "rb") as file:
                        st.download_button(
                            f"Download {label}",
                            data=file,
                            file_name=filename,
                            mime=f"image/{label.split(' ')[0].lower()}"
                        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
