# model_1-00

## About
The model is testing random masking on target with the context. With the model having the same architecture as 
`inpainting_mobidb\model_1-00` which had a simple architecture only including the the generator portion of the GAN. We 
tested this model to see how well the generator can generate disordered regions with random masking. 

## Results
With masking only 10% of the targets, we expect a higher accuracy because a lot of the information is given to the 
generator. But the metrics of the models, show that the model is not learning. The accuracy of the generator is at 9% 
which is similar to as good as guessing which amino acid for the disordered region. Looking at 
`inpainting_mobidb_1\model_aa_comp`, it shows that the amino acids wih the highest frequency overall (target and context)
are the ones being predicted showing that the generator is not learning during the training process and is not learning 
any significant features or organizing principles. 

## Conclusion
Try to see if the addition of a discriminator improves the generator in `inpainting_mobidb_1\model_0-02`. 
