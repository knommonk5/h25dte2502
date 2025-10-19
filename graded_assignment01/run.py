import os
import sys
import subprocess
import argparse
import torch


def run_training(dataset_size="small", epochs=60, batch_size=32):
    print("Running on CPU")
    device_type = "CPU"
    batch_size = "4"  #reduce batch for memory
    workers = "0"  #no parallel workers

    #dataset config
    if dataset_size == "small":
        dataset_path = "dte2502_ga01_small"  #small
    elif dataset_size == "large":
        dataset_path = "dte2502_ga01"  #full
    else:
        print(f"Unknown dataset size: {dataset_size}")
        return

    #command to execute main training script with parameters for the PHOSCnet model
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

    #training process
    try:
        #start, runs main.py
        process = subprocess.Popen(cmd)
        process.wait()

        #check success/fail
        if process.returncode == 0:
            print("Training completed successfully!")
            print("Model weights saved in: PHOSCnet_temporalpooling/")
        else:
            print(f"Training failed with exit code: {process.returncode}")

    except KeyboardInterrupt:
        #handle (Ctrl+C)
        print("Training interrupted by user")
    except Exception as e:
        #catch any unexpected errors
        print(f"Error during training: {e}")

def show_menu():
    #run menu
    print("\n" + "=" * 50)
    print("QUICK TRAINING:")
    print("1. Quick Small (10 epochs, batch=32)")
    print("2. Full Small (60 epochs, batch=32)")
    print("3. Extended Small (100 epochs, batch=32)")
    print("\nFULL TRAINING:")
    print("4. Quick Large (10 epochs, batch=16)")
    print("5. Full Large (30 epochs, batch=16)")
    print("6. Extended Large (60 epochs, batch=16)")
    print("\n9. Exit")
    print("-" * 50)


def main():
    #Handle scenarios
    parser = argparse.ArgumentParser(description='PHOSCnet GPU Training Runner')
    parser.add_argument('--scenario', type=int, help='Run specific scenario (1-9)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large'], help='Dataset size')

    args = parser.parse_args()

    while True:
        show_menu()
        try:
            #get input
            choice = input("Select option (1-6,9): ").strip()
            if choice == '9':
                print("Goodbye!")
                break
            elif choice in ['1', '2', '3', '4', '5', '6']:
                run_scenario(int(choice))
            else:
                print("Invalid choice. Please select 1-9.")
        except KeyboardInterrupt:
            #Handle Ctrl+C
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")



def run_scenario(choice, epochs=60, batch_size=32, dataset='small'):
    scenarios = {
        #small dataset training scenarios
        1: lambda: run_training("small", 10, 32),
        2: lambda: run_training("small", 60, 32),
        3: lambda: run_training("small", 100, 32),

        #large dataset training scenarios
        4: lambda: run_training("large", 10, 16),
        5: lambda: run_training("large", 30, 16),
        6: lambda: run_training("large", 60, 16),
    }

    #run scenario
    if choice in scenarios:
        print(f"Executing scenario {choice}...")
        scenarios[choice]()
    else:
        print("Invalid scenario")


if __name__ == "__main__":
    #avoid library conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()