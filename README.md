🎨 Photosculpt-web-image-filter-app

A Streamlit-based web application that lets you upload an image, apply various filters (like Gaussian Blur, Canny Edge Detection, Black & White, and Quality Adjustment), compare results side-by-side, and download the processed image.

---

 📖 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Available Filters](#available-filters)
* [Examples](#examples)
* [Dependencies](#dependencies)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

✨ Introduction

This app allows users to experiment with common image processing techniques through an intuitive web interface. It is built using [Streamlit](https://streamlit.io/) and [OpenCV](https://opencv.org/), enabling fast prototyping and easy deployment.

You can upload .jpg, .jpeg, or .png images, preview them alongside the filtered version, and download the result.

---

🚀 Features

* Upload and display your own image.
* Apply one of the following filters:

  * No Filter
  * Gaussian Blur (configurable kernel size & sigma)
  * Canny Edge Detection (configurable thresholds)
  * Black & White
  * JPEG Quality Adjustment
* Side-by-side comparison of original and processed images.
* Download the processed image in .jpeg or .png format.
* Reset the app to start over at any time.

---

🛠 Installation

⿡ Clone the repository:

git clone https://github.com/your-username/general-image-filter-app.git
cd general-image-filter-app


⿢ (Optional but recommended) Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


⿣ Install dependencies:

pip install -r requirements.txt


---

📋 Usage

Run the app using Streamlit:

streamlit run app.py


Then open the URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501)) in your browser.

---

🎨 Available Filters

| Filter Name              | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| *No Filter*            | Displays the original image                           |
| *Gaussian Blur*        | Blurs the image with configurable kernel size & sigma |
| *Canny Edge Detection* | Detects edges using the Canny algorithm               |
| *Black and White*      | Converts image to grayscale                           |
| *Quality Adjustment*   | Adjusts JPEG quality to reduce file size or quality   |

---

📷 Examples

| Original                                         | Gaussian Blur                                   | Canny Edge Detection                          |
| ------------------------------------------------ | ----------------------------------------------- | --------------------------------------------- |
| ![Original](https://via.placeholder.com/150x100) | ![Blurred](https://via.placeholder.com/150x100) | ![Canny](https://via.placeholder.com/150x100) |

---

📦 Dependencies

* [Streamlit](https://streamlit.io/) >= 1.0
* [OpenCV](https://opencv.org/) (cv2) >= 4.x
* numpy
* (Optional) Pillow (if Streamlit needs it internally)

You can install dependencies using:

pip install -r requirements.txt


Sample requirements.txt:

txt
streamlit
opencv-python
numpy


---

⚙ Configuration

* *MAX\_DISPLAY\_WIDTH* — Maximum width in pixels for displaying images in the app (600 by default).
* *MAX\_DOWNLOAD\_WIDTH* — Maximum width in pixels for downloadable images (1000 by default).

You can adjust these constants at the top of the app.py file if desired.

---

🐞 Troubleshooting

* If the app fails to start, ensure you have all dependencies installed.
* If OpenCV cannot open the uploaded file, make sure the file is a valid .jpg, .jpeg, or .png.
* For large images, processing may take a little longer.

---
