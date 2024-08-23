# Semantic Segmentation for Targeted Colorization

This project implements a deep learning model that combines semantic segmentation and colorization to selectively colorize portions of grayscale images.

## Requirements

See `requirements.txt` for a full list of dependencies. Key requirements include:

- Python 3.7+
- PyTorch 2.4.0
- torchvision 0.19.0
- numpy
- matplotlib
- Pillow
- scikit-learn
- opencv-python

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/semantic-colorization.git
   cd semantic-colorization
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Training the model:
   - Open and run `semantic_colorization_model.ipynb` in Jupyter Notebook or JupyterLab.
   - This notebook contains the model definition, training loop, and evaluation code.

2. Using the GUI:
   - Run the GUI script:
     ```
     python gui.py
     ```
   - Load an image, select regions to colorize by dragging the mouse, and click "Colorize" to see the result.

## Model Architecture

The model consists of two main components:
1. A semantic segmentation network (FCN-ResNet50)
2. A colorization network (custom CNN)

These components work together to produce targeted colorization based on semantic regions and user input.

## Performance

The model achieves the following performance metrics on the test set:

- Precision: 0.XXXX
- Recall: 0.XXXX

(Note: Replace X's with actual values after training and evaluation)

## Limitations and Future Work

- The current model is trained on a limited dataset and may not generalize well to all types of images.
- Future work could include fine-tuning on specific domains or incorporating more advanced colorization techniques.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
