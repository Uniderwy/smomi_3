# smomi_3


## Flip
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    return image,label
 ``` 
train - оранжевый
val - синий

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/flip.jpg) 
    
    
## Brightness/Contrast
```
def augment(image,label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_brightness(image, 0.7, seed=None)
    image = tf.image.random_contrast(image, 0.3, 1.3, seed=None)
    return image,label
 ```
### Brightness 0.1, Contract 0.7, 1.4
train - красный
val - голубой

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/bright3.jpg) 

### Brightness 0.1, Contract 0.2, 1.5
train - серый
val - оранжевый

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/bright1.jpg) 

### Brightness 0.7, Contract 0.3, 1.3
train - голубой
val - розовый

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/bright2.jpg) 
    
    
## Gauss noize
```
def augment(image,label):
    with tf.name_scope('Add_gaussian_noise'):
        noise_img = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.5, dtype=tf.float32)
        noise_img = tf.clip_by_value(noise_img, -1.0, 1.0)
    return noise_img,label
```
### Gauss stddev = 0.5
train - синий
val - красный

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/g1.jpg)    
   
### Gauss stddev = 0.25
train - красный
val - голубой

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/g2.jpg)  

### Gauss stddev = 0.1
train - оранжевый
val - синий

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/g3.jpg)    


## Rotate
```
def augment(image, label):
           degree = 10
           degrand = np.random.uniform(-degree, degree)
           image = tfa.image.rotate(image, np.pi * degrand / 180, interpolation='BILINEAR')
           return image, label
```   
### Rotate10
train - розовый
val - зеленый

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/rot10.jpg)  

### Rotate20
train - синий
val - красный

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/rot20.jpg) 

### Rotate30
train - голубой
val - розовый

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/rot30.jpg) 


## Optional
```      
def augment(image,label):
    with tf.name_scope('Add_gaussian_noise'):
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32)
        image = tf.clip_by_value(image, -1.0, 1.0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.7, seed=None)
    image = tf.image.random_contrast(image, 0.3, 1.3, seed=None)
    degree = 10
    degrand = np.random.uniform(-degree, degree)
    image = tfa.image.rotate(image, np.pi * degrand / 180, interpolation='BILINEAR')
    return image,label
```   
train - голубой
val - розовый

![alt text](https://github.com/Uniderwy/smomi_3/blob/main/opt1.jpg) 
  

  
  
    
    
    
