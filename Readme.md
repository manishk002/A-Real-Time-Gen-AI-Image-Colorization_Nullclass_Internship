# Advanced Image Colorization Project

This repository contains the code and documentation for an advanced image colorization project developed during an internship at NullClasses. The project focuses on three main tasks: hyperparameter tuning, semantic segmentation for targeted colorization, and interactive user-guided colorization.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Models](#models)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

This project explores advanced techniques in image colorization using deep learning. It consists of three main components:

1. **Hyperparameter Tuning**: Optimizing the colorization model's performance through systematic hyperparameter adjustment.
2. **Semantic Segmentation for Targeted Colorisation**: Implementing a model that combines semantic segmentation with colorization to selectively colorize specific regions of an image.
3. **Interactive User-Guided Colorization**: Developing a system that allows users to interactively control the colorization process by selecting regions and specifying colors.

## Installation

To set up the project environment:

1. Clone this repository:
   ```
   git clone https://github.com/your-username/advanced-image-colorization.git
   cd advanced-image-colorization
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Hyperparameter Tuning
To run the hyperparameter tuning experiment:
```
python hyperparameter_tuning.py
```

### Semantic Segmentation for Targeted Colorisation
To run the semantic segmentation-based colorization:
```
python semantic_colorization.py
```

### Interactive User-Guided Colorization
To launch the interactive colorization application:
```
python interactive_colorization.py
```

## Project Structure

```
advanced-image-colorization/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── colorization_model.py
│   └── segmentation_model.py
│
├── notebooks/
│   ├── hyperparameter_tuning.ipynb
│   ├── semantic_segmentation.ipynb
│   └── interactive_colorization.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── gui/
│   └── colorization_app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Models

The trained models can be found in the `models/` directory. Due to file size limitations, the model weights are hosted on Google Drive. You can download them using the following links:

- Colorization Model: [Download](https://drive.google.com/file/d/your-file-id/view?usp=sharing)
- Segmentation Model: [Download](https://drive.google.com/file/d/your-file-id/view?usp=sharing)

## Results

The project achieved the following results:

- Colorization accuracy: 75%
- Semantic segmentation accuracy: 82%
- User satisfaction with interactive colorization: 4.2/5

Detailed results and analysis can be found in the project report.

## Output Screens

Here are some screenshots demonstrating the Semantic Colorization application:

### Initial State
![Initial State of Semantic Colorization](https://github.com/manishk002/A-Real-Time-Gen-AI-Image-Colorization_Nullclass_Internship/blob/main/Images/input.png)
This image shows the initial state of the Semantic Colorization application. The interface includes a title bar with "Semantic Colorization" and two buttons at the bottom: "Load Image" and "Colorize".

### Colorized Output
![Colorized Output of Air Canada Plane](https://github.com/manishk002/A-Real-Time-Gen-AI-Image-Colorization_Nullclass_Internship/blob/main/Images/colorized.png)
This image displays a colorized output of an Air Canada airplane. The application has successfully loaded and processed an image of an Air Canada plane flying in a blue sky. The plane's details, including its livery and the sky's color, are clearly visible, demonstrating the effectiveness of the semantic colorization process.


## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
