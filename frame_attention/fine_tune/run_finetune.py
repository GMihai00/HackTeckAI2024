import torch
from ultralytics import YOLO

def main():
    # Load pre-trained model (path to the model you want to train/fine-tune)
    model = YOLO('yolo11n.pt')

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fine-tune the model
    results = model.train(data='data.yaml', epochs=50, imgsz=640, batch=16, device=0)

    # Save the fine-tuned model
    model.save('yolov11n_finetuned.pt')

if __name__ == '__main__':
    main()
