# grip-net
Real-time frictional Safety Margin Estimation with imaging RGB soft tactile fingertips. Implements a basic fully connected DNN with SOTA performance compared to CNN on training data, and which can outperform SOTA CNN in generalization. The DNN architecture and hyperparameters have not been fine-tuned. The input to the DNN are extracted dot cordinates and radii from preprocessed images. The preprocessing was done by Jingwen Tang, [here](https://github.com/Jin-2022-ha/tactile_images). The data courtesy of Jingwen Tang, [here](https://github.com/Jin-2022-ha/tactile_images).

### Description
![grip-net-scheme](https://github.com/lwelzel/grip-net/assets/29613344/d9910ecd-26ec-4221-8af8-b64d0196d837)
Nice figures are from R. Scharff et al. 2022, "Rapid manufacturing of color-based hemispherical soft tactile fingertips." DOI:10.1109/RoboSoft54090.2022.9762136

### Performance:
![image](https://github.com/lwelzel/grip-net/assets/29613344/cfffa97b-7486-4eab-b23a-f0547140af5a)
