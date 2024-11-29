import litmodels
import torch
from lightning.pytorch.demos.boring_classes import BoringModel

if __name__ == "__main__":
    # Define your model
    model = BoringModel()

    # Save the model's state dictionary
    torch.save(model.state_dict(), "./boring-checkpoint.pt")

    # Upload the model checkpoint
    litmodels.upload_model_files(
        "./boring-checkpoint.pt",
        "jirka/kaggle/boring-model",
    )

    # Download the model checkpoint
    model_path = litmodels.download_model_files("jirka/kaggle/boring-model", download_dir="./my-models")
    print(f"Model downloaded to {model_path}")

    # Load the model checkpoint
    loaded_model = BoringModel()
    loaded_model.load_state_dict(torch.load("./boring-checkpoint.pt"))
    print(loaded_model)
