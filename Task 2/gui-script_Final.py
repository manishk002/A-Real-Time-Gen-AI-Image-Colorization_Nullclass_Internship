import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the ColorizationNet
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.upsample(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.upsample(x)
        x = nn.functional.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        return x

# Define the SemanticColorizationModel
class SemanticColorizationModel(nn.Module):
    def __init__(self):
        super(SemanticColorizationModel, self).__init__()
        self.segmentation_model = fcn_resnet50(pretrained=True)
        self.colorization_model = ColorizationNet()

    def forward(self, x, mask=None):
        # Semantic segmentation
        seg_output = self.segmentation_model(x)['out']
        seg_mask = torch.argmax(seg_output, dim=1, keepdim=True)
        
        # Apply user-defined mask if provided
        if mask is not None:
            seg_mask = seg_mask * mask
        
        # Convert input to grayscale
        gray_input = torch.mean(x, dim=1, keepdim=True)
        
        # Colorization
        color_output = self.colorization_model(gray_input)
        
        # Resize color_output to match gray_input size
        color_output_resized = F.interpolate(color_output, size=gray_input.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine original grayscale with colorized output
        combined_output = torch.cat([gray_input, color_output_resized], dim=1)
        
        # Apply segmentation mask
        masked_output = combined_output * seg_mask
        
        return masked_output, seg_mask


# Load the trained model
model = SemanticColorizationModel().to(device)
# Assuming state_dict is loaded from a file
state_dict = torch.load('/Users/manish/Downloads/colorization_model.pth', map_location=torch.device('cpu'))

# Initialize your model
model = SemanticColorizationModel()

# Load state dict with strict=False to ignore non-matching keys
model.load_state_dict(state_dict, strict=False)

# Now you can use the model
model.eval()


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ColorizeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Semantic Colorization")

        self.canvas = tk.Canvas(master, width=512, height=512)
        self.canvas.pack()

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.colorize_button = tk.Button(master, text="Colorize", command=self.colorize)
        self.colorize_button.pack()

        self.image = None
        self.mask = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert('RGB')
            self.image = self.image.resize((512, 512))
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.mask = np.zeros((512, 512), dtype=np.uint8)

    def on_mouse_drag(self, event):
        if self.image:
            x, y = event.x, event.y
            cv2.circle(self.mask, (x, y), 10, 255, -1)
            self.update_preview()

    def update_preview(self):
        preview = cv2.addWeighted(np.array(self.image), 0.7, cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
        preview = Image.fromarray(preview)
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def colorize(self):
        if self.image and self.mask is not None:
            # Prepare input
            input_tensor = transform(self.image).unsqueeze(0)
            mask_tensor = torch.from_numpy(self.mask).unsqueeze(0).unsqueeze(0).float() / 255.0

            # Generate output
            with torch.no_grad():
                output, _ = model(input_tensor, mask_tensor)

            # Convert output to image
            output_image = output.squeeze().permute(1, 2, 0).numpy()
            output_image = (output_image * 0.5 + 0.5).clip(0, 1)
            output_image = Image.fromarray((output_image * 255).astype(np.uint8))

            # Display result
            self.photo = ImageTk.PhotoImage(output_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

root = tk.Tk()
gui = ColorizeGUI(root)
root.mainloop()
