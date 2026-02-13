import yaml

from adni_mri_classification.data.datasets import ADNIDataset

def main(path="./test.yaml"):

    # Read configuration file
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Print configuration
    print(cfg)

    # Load dataset
    dataset = ADNIDataset(**cfg["data"])

    # Perform dataset splitting
    

    pass
    

if __name__ == "__main__":
    
    main()