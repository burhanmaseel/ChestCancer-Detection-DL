import argparse
import kagglehub
from models import basic_cnn, vgg16_transfer

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate CT scan classification models')
    parser.add_argument('--model', type=str, choices=['cnn', 'vgg16', 'both'], default='both',
                      help='Model to train: cnn, vgg16, or both')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for CNN training')
    parser.add_argument('--initial-epochs', type=int, default=15, help='Number of initial epochs for VGG16 training')
    parser.add_argument('--fine-tune-epochs', type=int, default=15, help='Number of fine-tuning epochs for VGG16')
    args = parser.parse_args()

    # Download and setup dataset
    print("Downloading Dataset from KaggleHub!")
    path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")
    print("Dataset Downloaded!")

    # Setup data directories
    data_dir_train = f"{path}/Data/train"
    data_dir_val = f"{path}/Data/valid"
    data_dir_test = f"{path}/Data/test"

    if args.model in ['cnn', 'both']:
        print("\nBasic CNN Model")
        history_cnn, model_cnn = basic_cnn.train_and_evaluate(
            data_dir_train, data_dir_val, data_dir_test,
            img_size=256,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        basic_cnn.plot_training_history(history_cnn)

    if args.model in ['vgg16', 'both']:
        print("\nVGG16 Transfer Learning Model")
        history_frozen, history_finetune, model_vgg16 = vgg16_transfer.train_and_evaluate(
            data_dir_train, data_dir_val, data_dir_test,
            img_size=224,
            batch_size=args.batch_size,
            initial_epochs=args.initial_epochs,
            fine_tune_epochs=args.fine_tune_epochs
        )
        vgg16_transfer.plot_training_history(history_frozen, history_finetune)

if __name__ == '__main__':
    main()
