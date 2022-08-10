# model 0-0

## About the Model
This is the preliminary GAN model that will the base for future networks. The model architecture is based off of Li et al (1). 


## Results

### Accuracy
The training and accuracy loss track each other but the validation accuract is overall better than the training, indicating 
that the model is learning from training. The ending accuracy for the validation and training datasets are not very different 
with a training accuracy of 5.4% and a validation accuracy of 5.5%.

  
![Alt text](out/metrics_accuracy_model0-0.png)

### Losses
The loss curves are very sporadic with values jumping between 0 to 7.5 in the first 300 epochs for the discriminator. 
During this period the generator overall has higher loss values than the discriminator. The generator has loss values 
that range from 2.5 to 17.5. From 300 to 450 epochs, the discriminator has higher loss values than generator with values 
ranging from 6 to 10 and the generator values ranging from 1 to 2.5. From 450 to 500 epochs, there are no drastic changes with 
generator values ranging in a max difference of 1 and the same with the generator. 

![Alt text](out/metrics_loss_model0-0.png)

## Conclusions
The model did not converge when therefore looked into diagnosing model architecture problems and deduced that there needs
to be a simpler model architecture based off of our size of data. 

[Notes on Diagnosing GAN Training issues](https://coconut-honeycup-be9.notion.site/Fixing-GAN-Training-4b02b3f5e15847e2bb9a9d302e0e89af)

##References
(1) DOI: 10.1109/ICTAI.2017.00166