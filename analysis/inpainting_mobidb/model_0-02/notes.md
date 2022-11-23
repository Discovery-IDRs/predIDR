# model_0-02

# About the Model
Applying alternate training on the simplified network architecture in model_0-01. The alternate training will be training 
the generator every 10 batches and the discriminator for every batch. 

# Results

## Accuracy
The accuracy for the validation and training datasets are very similar after 300 epochs, both around 5.5%. The accuracy seems 
to be more stable than tha last models with a less steep increases and drops. Hitting a steady 5% accuracy indicates
probable mode collapse, where the most common acid is being predicted each time.

![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_0-02/out/metrics_accuracy_model0-02.png)

## Losses 
The loss curves look a lot better with all increases being gradual and no steep increases and drops. There is no large 
difference in the values for the losses for the generator or discriminator, all being in the range of 0.5 to 1.8. 
There is very little difference for the discriminator for the training and validation dataset but for the generator there 
is a decrease in the loss for the validation compared to the training loss. 

![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_0-02/out/metrics_loss_model0-02.png)

## Model Outputs
The amino acid composition of the disordered regions for a model with a simpler architecture and alternate training 
has contiguous repeated regions of the amino acid Glycine(G) and then various amino acids towards the ends of the
disorder regions. 

# Conclusion
The steadiness of the accuracy and loss curves show that alternate training and a simplified architecture works better 
with the model and our dataset. We will be looking into our model outputs to determine if there is mode collpase. 