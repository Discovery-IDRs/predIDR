# model_1-00

# About the Model
This model is only the generative portion of model_0-01. It was tested to determine if a simpler training task would be 
more successful at reproducing the targets.

# Results

## Accuracy
There is overall less large oscillations for the accuracy compared to `model0-01` showing that a simpler architecture
creates a more stable network that is able to learn. The validation and training accuracy curves follow each other 
with both of them ending with an accuracy about 7.5%. The accuracy is low showing tha the the model is not correctly
predicting the disordered regions. 

![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_1-00/out/metrics_accuracy_model1-00.png)


## Losses
The loss curve is much more stable that previous loss curves showing tha the model is learning from it's losses with a 
steep decrease at 10 epochs and continues at a low loss. 
![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_1-00/out/metrics_loss_model1-00.png)

## Model Outputs
There is continued contiguous regions in the disordered regions with the amino acid composition being mostly comprised of
repeated segments of the amino acid Glutamic Acid(E).

# Conclusion 
Because of the continued model outputs of repeated contiguius region of the same amino acid, this leads to the suspicion 
of needing to change the structure of the data rather than the structure of the model. 
