# model_0-02

## About the Model
The model is testing random masking on target with the context. With the model having the same architecture as 
`inpainting_mobidb\model_0-02` which had a simple architecture with the generator and discriminator of the GAN.

# Results

After seeing the results of the generator alone with random masking of the disordered amino acids, it was believed that
there was not enough information given to the generator to predict what a disordered region is composed of because none 
of the disordered region is given to the model. 

With the generator only in `inpainting_mobidb_1\model_0-02` there was an accuracy of about 9% and when comparing with 
`inpainting_mobidb_1\model_1-00` to see if the discriminator's decisions would improve the accuracy of the generator,
the accuracy also does not go above 9% this shows tha the model of having the discriminator probably does not yield any 
significant information for the generator to generate a better disordered region.

When looking at the model outputs, the generator predicts the most common amino acid to yield the highest accuracy 
instead of learning what a disordered region is. 

When the model only has 10% masking, the generator still leads only to an accuracy of 9% showing that the model is not
learning during training and instead is in mode collapse. 

# Conclusion
Need to reframe model. The model is not achieving an accuracy above 10% even when given most of the disordered region so 
model is not learning. Possible new models is instead of giving the masked target with surrounding context is to give 
random noise to the generator as traditionally many GANs are structured. 