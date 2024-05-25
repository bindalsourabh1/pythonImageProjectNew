import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess the image for OCR.

    Parameters:
    image (ndarray): The input image.

    Returns:
    ndarray: The preprocessed image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return thresh


def extract_text(image):
    """
    Extract text from the image using Tesseract OCR.

    Parameters:
    image (ndarray): The input image.

    Returns:
    str: The extracted text.
    """
    # Use Tesseract to extract text
    text = pytesseract.image_to_string(image)

    return text


def segment_elements(image):
    """
    Segment visual elements in the image.

    Parameters:
    image (ndarray): The input image.

    Returns:
    list: List of segmented elements.
    """
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and return the segmented elements
    segmented_elements = []
    for cnt in contours:
        # Apply filtering criteria to select relevant elements
        if cv2.contourArea(cnt) > 100:
            segmented_elements.append(cnt)

    return segmented_elements


def main():
    # Load the image
    image = cv2.imread('image.png')

    # Preprocess the image for OCR
    preprocessed_image = preprocess_image(image)

    # Extract text using Tesseract OCR
    extracted_text = extract_text(preprocessed_image)
    print("Extracted Text:")
    print(extracted_text)

    # Segment visual elements
    segmented_elements = segment_elements(preprocessed_image)

    # Draw contours around segmented elements
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, segmented_elements, -1, (0, 255, 0), 2)

    # Display the image with contours
    cv2.imshow('Segmented Elements', image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
