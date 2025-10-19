import os
import sys
import subprocess
import torch


def run_training(dataset_size="small", epochs=60, batch_size=32):
    print(f"Starting GPU Training on {dataset_size.upper()} Dataset...")
    print("=" * 50)

    #GPU detection and optimize
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        device_type = "GPU"
        workers = "4"
    else:
        print("No GPU *******************************")
        device_type = "CPU"
        batch_size = "4"  #
        workers = "0"  #no parallel workers

    #dataset configuration
    if dataset_size == "small":
        dataset_path = "dte2502_ga01_small"  #small dataset
    elif dataset_size == "large":
        dataset_path = "dte2502_ga01"  #full dataset
    else:
        print(f"Unknown dataset size: {dataset_size}")
        return

    #construct training command
    cmd = [
        sys.executable, "main_gpu.py",
        "--mode", "train",
        "--model", "PHOSCnet_temporalpooling",
        "--train_csv", f"{dataset_path}/train.csv",
        "--train_folder", f"{dataset_path}/train",
        "--valid_csv", f"{dataset_path}/valid.csv",
        "--valid_folder", f"{dataset_path}/valid",
        "--batch_size", str(batch_size),
        "--num_workers", workers,
        "--epochs", str(epochs)
    ]

    # Display training configuration
    print(f"Training Configuration:")
    print(f"   Device: {device_type}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Workers: {workers}")
    print("-" * 50)

    #run training
    try:
        process = subprocess.Popen(cmd)
        process.wait()

        if process.returncode == 0:
            print("Training completed successfully!")
        else:
            print(f"Training failed with exit code: {process.returncode}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")

def show_menu():
    #Menu
    print("\n" + "=" * 50)
    print("QUICK TRAINING (Debugging):")
    print("1. Quick Small Dataset Training (10 epochs, batch=32)")
    print("2. Full Small Dataset Training (60 epochs, batch=32)")
    print("3. Extended Small Dataset Training (100 epochs, batch=32)")
    print("\nFULL TRAINING (Production):")
    print("4. Quick Large Dataset Training (10 epochs, batch=16)")
    print("5. Full Large Dataset Training (30 epochs, batch=16)")
    print("6. Extended Large Dataset Training (60 epochs, batch=16)")
    print("\n9. Exit")
    print("-" * 50)


def main():
    #menu loop
    while True:
        show_menu()
        try:
            choice = input("Select option (1-6,9): ").strip()
            if choice == '9':
                print("Goodbye!")
                break
            elif choice in ['1', '2', '3', '4', '5', '6']:
                run_scenario(int(choice))
            else:
                print("Invalid choice. Please select 1-6,9.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_scenario(choice, epochs=60, batch_size=32, dataset='small'):
    #scenario menu
    scenarios = {
        #small dataset scenarios
        1: lambda: run_training("small", 10, 32),  # Quick debug
        2: lambda: run_training("small", 60, 32),  # Standard training
        3: lambda: run_training("small", 100, 32),  # Extended training

        #large dataset scenarios (reduced batch_size for larger images)
        4: lambda: run_training("large", 10, 16),
        5: lambda: run_training("large", 30, 16),
        6: lambda: run_training("large", 60, 16),
    }

    if choice in scenarios:
        scenarios[choice]()
    else:
        print("Invalid scenario")


if __name__ == "__main__":
    #avoid library conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()