# grip-net
Real-time frictional Safety Margin Estimation with imaging RGB soft tactile fingertips. Implements a basic fully connected DNN with SOTA performance compared to CNN on training data, and which can outperform SOTA CNN in generalization. The DNN architecture and hyperparameters have not been fine-tuned. The input to the DNN are extracted dot cordinates and radii from preprocessed images. The preprocessing was done by Jingwen Tang, [here](https://github.com/Jin-2022-ha/tactile_images). The data courtesy of Jingwen Tang, [here](https://github.com/Jin-2022-ha/tactile_images).

Pre-trained models are available as [pytorch lightning checkpoints](https://drive.google.com/file/d/1KeKmGuuRaWEkp7NWl-InMHoINmz16YYb/view?usp=drive_link), and [pytorch state dictionaries](https://drive.google.com/file/d/1-h-maq0fqgntdyyyTpQaWdCIk8ypTRB6/view?usp=drive_link). Normalized data is available as [.csv](https://drive.google.com/file/d/1JZBg0QfTN5XQhWx3Feh0RL_YtRlmk_wD/view?usp=drive_link).

### Description
![grip-net-scheme](https://github.com/lwelzel/grip-net/assets/29613344/d9910ecd-26ec-4221-8af8-b64d0196d837)
Nice figures are from R. Scharff et al. 2022, "Rapid manufacturing of color-based hemispherical soft tactile fingertips." DOI:10.1109/RoboSoft54090.2022.9762136

### Performance:
![image](https://github.com/lwelzel/grip-net/assets/29613344/cfffa97b-7486-4eab-b23a-f0547140af5a)
