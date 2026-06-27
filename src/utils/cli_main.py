from lightning.pytorch.cli import LightningCLI

def main():
    # Usage:
    #   python main.py fit      --config configs/config_fit.yaml
    #   python main.py test     --config configs/config_test.yaml
    #   python main.py predict  --config configs/config_predict.yaml
    LightningCLI(
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()