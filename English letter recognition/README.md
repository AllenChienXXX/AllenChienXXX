# Hi my name is Allen, welcome to my ML project! :muscle:
This project is all written by myself,with some help by my father.
## Basic about this project
This project is about CNN, which is convolution neural network.
To understand it, you need to know how machine learning works, I didn't use any machine learning module(Tensorflow,keras,etc...)
But I read a lot of books about deep learning,so I'll be able to start it at the bottom.
I'll not explain the algorithms & details about how layers work here.
## Code
:point_right:[Here](https://github.com/AllenChienXXX/Projects-Practices/blob/projects/English%20letter%20recognition/Code.py)
## datasets
:point_right:[Training](https://drive.google.com/drive/folders/1xpOHmM0b1437qn1lOcvwam7ko_tSBPUP?usp=sharing)


:point_right:[Testing](https://drive.google.com/drive/folders/1L4EludX9aqF6yVwJBBqNYmqYhlLf4yUU?usp=sharing)

## Layers

>  Convolution --> Dropout --> Pooling --> Dense --> Dropout --> Output 

## Output
![image](https://github.com/AllenChienXXX/Projects-Practices/blob/projects/English%20letter%20recognition/Output.png)
## How to use the code 
- Download the dataset
- Inside the code you will see how I upload data and shaping it
## Problems I encounter
After I use the original data to train & test,the train error is nearly 100%,but the test error is very low,it seems to be a overfitting problem,so I try to change the amount of layers and put in some dropout layer,but that didn't change the result,but then I put both train and test dataset together and split it again evenly,and that did work!I think the main reason is because I use Adagrad and the weights and bias are able to recognize "Z" at the end,so it might not be able to recognize "A" again,the another way to solve this using "Adam" as optimizer:eyes:,but I can't write it in code,maybe I will try next time!


## Contact me
- :point_right:allen71090@gmail.com:eyes: 


# That's all ! :raised_hands: Hope I will join you and do some projects together!
