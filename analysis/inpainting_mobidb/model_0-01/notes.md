# model_0-01

## About the Model
Simplified GAN architecture from model_0_0 because we believed that based off of the size of our data, our architecture 
was too big. Motivations to simplify the architecture also include that the model was poorly constrained and losses weren't
large enough to reach a local minimum. 

**Architecture:**
* _Generator_:  4 convolutional (2-4-8-16) and 3 deconvolutional (8-4-2), stride = 1
* _Discriminator_: 3 convolutional (8-4-2) and 1 flattened layer, stride = 2

## Results 
### Accuracy
There is some improvement of the validation accuracy from training but it seems that it behave similarly as it did in the 
training and the ending accuracy for validation dataset is 10% and the accuracy for the training dataset is 8.9%. The accuracy
has steep drop offs throughout the epochs. We can deduce that the model is not learning anything and is essentially guessing 
the amino acid composition of disordered sequences. 

![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_0-01/out/metrics_accuracy_model0-01.png)
### Losses
The generator loss stays constant with a value of 0.5-2, while the loss of the discriminator has much larger variance, bouncing around
values to 2 to 16. The loss curves look better than model_0-0 because there is less sporadic drops and rises, showing that the model works better
with a simplified architecture. 

![Alt text](/Users/SamihaMahin/PycharmProjects/predIDR/analysis/inpainting_mobidb/model_0-01/out/metrics_loss_model0-01.png)

### Model Outputs
Looking at the training of model outputs for a simpler architecture, all disordered sequences look similar for all contexts and have
repeated segments of the amino acid isoleucine(I). 

## Conclusion
The loss curve looks better than model_0-0 because of the lack of sporadic oscillation. We will be implementing alternate 
training where on this simplified model architecture for model_0-02.