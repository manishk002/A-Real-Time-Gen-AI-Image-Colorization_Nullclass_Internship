# Image Colorization with Hyperparameter Tuning

This project implements an image colorization model using PyTorch, with hyperparameter tuning to improve model performance.

## Requirements

Install the required packages using:

```
pip install -r requirements.txt
```

## Usage

1. Open and run the `colorization_model_training.ipynb` notebook in a Jupyter environment.
2. The notebook will perform hyperparameter tuning and train the final model.
3. The best model weights will be saved as 'best_model.pth'.
4. The final model trained with the best hyperparameters will be saved as 'final_model.pth'.

## Model Weights

Due to file size limitations on GitHub, the model weights are hosted on Google Drive:

- [Best Model Weights](https://drive.google.com/file/d/...)
- [Final Model Weights](https://drive.google.com/file/d/...)

Download these files and place them in the same directory as the notebook to use the trained models.

## Results

The notebook includes a visualization function to display the original, grayscale, and colorized images. Run this function at the end of the notebook to see the results of the colorization model.

## Performance

The model's performance is evaluated using Mean Squared Error (MSE) loss. The hyperparameter tuning process aims to minimize this loss. Refer to the notebook output for detailed performance metrics.

